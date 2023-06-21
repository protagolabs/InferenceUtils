import functools
import json
import os
import shutil
import signal
import tarfile
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Union, Callable, Optional

import boto3
import requests
from botocore.exceptions import ClientError, BotoCoreError

from inferenceUtils.fileDownloader import FileDownloadConfig, FileDownloader
from inferenceUtils.logSetting import logger
from inferenceUtils.redisClient import RedisClient

stream_name = os.getenv("STREAM_NAME", "DefaultStreamName")

model_bucket_name = os.getenv("MODEL_BUCKET_NAME", "netmind-inference-model-bucket")

report_endpoint = os.getenv("REPORT_ENDPOINT", "http://localhost:8080/report")

HOME_DIR = os.path.expanduser("~")
DEFAULT_CACHE_DIR = os.getenv("DEFAULT_CACHE_DIR", os.path.join(HOME_DIR, ".cache/netmind/models"))
env = os.getenv("ENV", "dev").lower()


class Utils(object):

    def __init__(self):
        self.redis_client = RedisClient()
        self.s3_client = boto3.Session(region_name="us-west-2").client("s3")
        # self.sns_client = boto3.client("sns")
        self.sqs_client = boto3.client('sqs', region_name='us-west-2')
        self.stream_name = stream_name
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _sigterm_handler(self, signum, frame):
        logger.info("Received SIGTERM signal, shutting down gracefully...")
        self.should_stop = True

    def run(self, execute_fuc, interval=0):
        assert isinstance(execute_fuc, Callable), "execute_fuc must be a function or lambda"
        res = requests.post(
            f"{report_endpoint}/inform_ready", json={"stream_name": stream_name})
        logger.info(f"called inform_ready rep:{res.status_code}；"
                    f" now start to monitor stream:{stream_name} with interval:{interval}")
        while not self.should_stop:
            execute_fuc()
            time.sleep(interval)
        self.redis_client.close()
        logger.info("worker pod gracefully stopped.")

    def download_and_extract(self, object_key: str) -> str:
        s3 = boto3.client('s3')
        tmpdir = tempfile.mkdtemp()
        file_path = os.path.join(tmpdir, object_key.split('/')[-1])
        s3.download_file(model_bucket_name, object_key, file_path)

        self.extract_file(file_path, tmpdir)
        return tmpdir

    def extract_file(self, file_path, tmpdir):
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

    def download_file_chunk(self, bucket_name, object_key, start_byte, end_byte, output_filename):
        range_header = f'bytes={start_byte}-{end_byte}'
        response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key, Range=range_header)

        with open(output_filename, 'wb') as f:
            for chunk in response['Body'].iter_chunks():
                f.write(chunk)

    def concurrent_download(self, object_key, num_proc=1, merge_block_size=1024 * 64):
        exist, extract_path = self.check_file_exist(object_key)
        print("debug folder exist:", exist, "extracted_path:", extract_path)
        if not exist:
            config = FileDownloadConfig(
                cache_dir=DEFAULT_CACHE_DIR,
                timeout=600,
                num_proc=num_proc,
                merge_block=merge_block_size
            )
            downloader = FileDownloader(config)
            file_path = downloader.download_to_temp_file(bucket=model_bucket_name, key=object_key)
            extract_path = downloader.extract_archive(file_path)
        return extract_path

    def check_file_exist(self, object_key):
        file_name_without_ext, ext = os.path.splitext(object_key)
        folder = os.path.join(DEFAULT_CACHE_DIR, file_name_without_ext)
        if os.path.exists(os.path.join(folder, "flag.conf")) and os.path.isdir(folder):
            return True, folder
        else:
            return False, ""

    def get_input_parameters(self, batch=1, is_first_model=True, json_decode=False) -> List[Dict[str, any]]:
        """
        param:
            batch: fetch max batch size from stream
            is_first_model: set message_id into input dict if true, so that it can be used to response
                            set it to false if the model is not the first model in the pipeline
            json_decode: decode body from json to dict; required if url is passed in json body,
                        so that utils can download

        :return:
        return a list of input parameters-dict, include "id"(need to be set when response) and "body"(json input/ dict)
        example [{"id": "24f2f0ba-5a8c-4625-be37-d7bf3fc46e09", "body": '{"text": "hello"}'}]
        """
        max_retry_times = 10
        start_time = time.time()
        for i in range(max_retry_times):
            messages = self.redis_client.get_data_from_stream(count=batch)
            input_list = []
            try:
                # messages = [[streamName, [(message_id, {message_body}))}]]]
                for message in messages[0][1]:
                    input_dict = message[1]
                    if is_first_model and "id" in input_dict.keys():
                        logger.warning("input dict already has id, which means it may not be the first model in "
                                       "pipeline we will ignore is_first_model parameters ")
                    if is_first_model and "id" not in input_dict.keys():
                        input_dict.update({"id": message[0]})
                    if self._is_fresh(input_dict.pop("recordTime")):
                        if json_decode:
                            input_dict = self.message_decode(input_dict)
                        input_list.append(input_dict)
                if not input_list:
                    continue
                print("get input", input_list, "time used=", time.time() - start_time)
                return input_list
            except IndexError as e:
                logger.error(f"exception happened when get input :{e}")
                return []

    def message_decode(self, input_dict):
        input_dict["body"] = json.loads(input_dict["body"])
        for key, value in input_dict["body"].items():
            if value.startswith("http"):
                content = self.download_file(value)
                input_dict["body"][key] = content
        return input_dict

    def download_file(self, url) -> Optional[bytes]:
        response = requests.get(url)
        if 300 > response.status_code >= 200:
            content_type = response.headers['content-type']
            if 'image' in content_type or 'audio' in content_type:
                file_data = response.content
                return file_data
            else:
                logger.error("try to download image or audio file, but the content-type is %s", content_type)
                return None
        else:
            logger.error("download file error, status_code: %s", response.status_code)
            return None

    # set message back to redis stream
    def set_task_back(self, message_with_id: List[Dict[str, str]], next_id):
        for message_dict in message_with_id:
            message_id = next(iter(message_dict))
            self.redis_client.set_message_to_next_stream(next_id, message_id, message_dict.get(message_id))

    def report_response(self, message_with_id: List[Dict[str, str]], use_muti_thread=False):

        def response_and_ack():
            for message_dict in message_with_id:
                message_id = next(iter(message_dict))
                self.redis_client.set_json_response_to_list(message_id, message_dict.get(message_id))
                self.redis_client.ack_message(message_id)

        partial_inner_closure = functools.partial(response_and_ack)
        if use_muti_thread:
            self.executor.submit(partial_inner_closure)
        else:
            response_and_ack()

    def record_response_async(self, message_with_id: List[Dict[str, Union[str, dict]]], use_muti_thread=False):

        def send_message_to_sns():
            for message_dict in message_with_id:
                message_id = next(iter(message_dict))
                self._send_message_to_sqs(message_id, message_dict.get(message_id))

        partial_inner_closure = functools.partial(send_message_to_sns)
        if use_muti_thread:
            self.executor.submit(partial_inner_closure)
        else:
            send_message_to_sns()

    # def _send_message_to_sns(self, sns_topic, message: dict):
    #     try:
    #         message_json = json.dumps({"default": json.dumps(message)})
    #         topic_arn = f"arn:aws:sns:us-west-2:134622832812:{sns_topic}"
    #         response = self.sns_client.publish(
    #             TopicArn=topic_arn, Message=message_json, MessageStructure="json"
    #         )
    #         print(f"Published message to {sns_topic}.")
    #         return response["MessageId"]
    #
    #     except ClientError:
    #         print(f"[Warning] Couldn't publish message to {sns_topic} with body {message}.")

    def _send_message_to_sqs(self, message_id, message_body):
        message_id = message_id.split("__")[0]
        endpoint_unique_id = stream_name.split("__")[0]
        async_queue_url = f"https://sqs.us-east-1.amazonaws.com/134622832812/inference-message-async-{env}" \
                          f"-{endpoint_unique_id}"
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
            logger.info(f"already send message to sqs {response}")
            return response
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error sending message to SQS: {str(e)}")

    def inform_ready(self):
        pass

    @staticmethod
    def _is_fresh(start_time, timeout=10):
        if time.time() - float(start_time) > timeout:
            logger.warning(" message timeout over 10s")
            return False
        else:
            return True


if __name__ == '__main__':
    utils = Utils()
    print(utils.get_input_parameters())
