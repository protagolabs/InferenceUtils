import json
import os
import shutil
import signal
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Union

import boto3
import requests
from botocore.exceptions import ClientError, BotoCoreError

from inferenceUtils.redisClient import RedisClient

stream_name = os.getenv("STREAM_NAME", "DefaultStreamName")

model_bucket_name = os.getenv("MODEL_BUCKET_NAME", "netmind-inference-model-bucket")

report_endpoint = os.getenv("REPORT_ENDPOINT", "http://localhost:8080/report")


class Utils(object):

    def __init__(self):
        self.redis_client = RedisClient()
        self.s3_client = boto3.Session(region_name="us-west-2").client("s3")
        self.sns_client = boto3.client("sns")
        self.sqs_client = boto3.client('sqs', region_name='us-west-2')
        self.stream_name = stream_name
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def _sigterm_handler(self, signum, frame):
        print("Received SIGTERM signal, shutting down gracefully...")
        self.should_stop = True

    def run(self, execute_fuc, interval=0):
        res = requests.post(
            f"{report_endpoint}/inform_ready", json={"endpoint_id": stream_name})
        print(
            f"called inform_ready rep:{res.status_code}； now start to monitor stream:{stream_name} with interval:{interval}")
        while not self.should_stop:
            execute_fuc()
            time.sleep(interval)
        self.redis_client.close()
        print("worker pod gracefully stopped.")

    def download_and_extract(self, object_key: str) -> str:
        s3 = boto3.client('s3')
        tmpdir = tempfile.mkdtemp()
        file_path = os.path.join(tmpdir, object_key.split('/')[-1])
        s3.download_file(model_bucket_name, object_key, file_path)

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(tmpdir)
        else:
            raise ValueError('Unsupported compression format')
        folder = Path(tmpdir)
        dir_list = [item for item in folder.iterdir() if item.is_dir() and item.name != "__MACOSX"]
        if len(dir_list) == 1 and len(list(folder.iterdir())) == 2:
            print("has sub folder")
            for sub_item in dir_list[0].iterdir():
                shutil.move(str(sub_item), str(folder))
        return tmpdir

    def get_input_parameters(self, batch=1) -> List[Dict[str, any]]:
        """
        :return:
        return a list of input parameters-dict, include "id"(need to be set when response) and "body"(json input)
        example [{"id": "24f2f0ba-5a8c-4625-be37-d7bf3fc46e09", "body": '{"text": "hello"}'}]
        """
        max_retry_times = 10
        for i in range(max_retry_times):
            messages = self.redis_client.get_data_from_stream(count=batch)
            input_list = []
            try:
                # messages = [[streamName, [(message_id, {message_body}))}]]]
                for message in messages[0][1]:
                    input_dict = message[1]
                    input_dict.update({"id": message[0]})
                    if self._is_timeout(input_dict.pop("recordTime")):
                        input_list.append(input_dict)
                if not input_list:
                    continue
                print("get input", input_list)
                return input_list
            except IndexError as e:
                print("exception happened when get input", e)
                return []

    # def report_response(self, message_id_list: List[str], **kwargs):
    #     for message_id in message_id_list:
    #         self.redis_client.set_response_to_list(message_id, **kwargs)
    #     self.redis_client.ack_message(message_id_list)

    def report_response(self, message_with_id: List[Dict[str, str]]):
        for message_dict in message_with_id:
            message_id = next(iter(message_dict))
            self.redis_client.set_json_response_to_list(message_id, message_dict.get(message_id))
            self.redis_client.ack_message(message_id)

    def record_response_async(self, message_with_id: List[Dict[str, Union[str, dict]]]):
        for message_dict in message_with_id:
            message_id = next(iter(message_dict))
            self._send_message_to_sqs(message_dict.get(message_id), message_id)

    def _send_message_to_sns(self, sns_topic, message: dict):
        try:
            message_json = json.dumps({"default": json.dumps(message)})
            topic_arn = f"arn:aws:sns:us-west-2:134622832812:{sns_topic}"
            response = self.sns_client.publish(
                TopicArn=topic_arn, Message=message_json, MessageStructure="json"
            )
            print(f"Published message to {sns_topic}.")
            return response["MessageId"]

        except ClientError:
            print(f"[Warning] Couldn't publish message to {sns_topic} with body {message}.")

    def _send_message_to_sqs(self, message_body, message_id):
        message_id = message_id.split("__")[0]
        async_queue_url = f"https://sqs.us-east-1.amazonaws.com/134622832812/inference-message-async-{message_id}"
        if isinstance(message_body, str):
            message_body = json.loads(message_body)
        result_message = {
            "response_body_json": message_body,
            "request_id": message_id,
            "request_body_json": "",
            "description": "succeed"
        }

        try:
            response = self.sqs_client.send_message(
                QueueUrl=async_queue_url,
                MessageBody=json.dumps(result_message)
            )
            print(f"already send message to sqs {response}")
            return response
        except (BotoCoreError, ClientError) as e:
            print(f"Error sending message to SQS: {str(e)}")

    def inform_ready(self):
        pass

    @staticmethod
    def _is_timeout(start_time, timeout=10):
        if time.time() - float(start_time) > timeout:
            print("timeout over 10s")
            return False
        else:
            return True


if __name__ == '__main__':
    utils = Utils()
    print(utils.get_input_parameters())
