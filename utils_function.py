import json
import os
import tempfile
import time
from typing import List, Dict

import boto3

from redisClient import RedisClient

stream_name = os.getenv("STREAM_NAME", "DefaultStreamName")

model_bucket_name = os.getenv("MODEL_BUCKET_NAME", "DefaultModelBucketName")


class Utils(object):

    def __init__(self):
        self.redis_client = RedisClient()
        self.s3_client = boto3.Session(region_name="us-west-2").client("s3")
        self.stream_name = stream_name

    def download_model_and_unzip(self, model_id, use_auth_token=False) -> str:
        tf = tempfile.NamedTemporaryFile(mode="w+b")
        self.s3_client.download_fileobj(model_bucket_name, model_id, tf)
        tf.seek(0)
        return tf.name

    def get_input_parameters(self, batch=1) -> List[Dict[str, any]]:
        messages = self.redis_client.get_data_from_stream(count=batch)
        input_list = []
        try:
            for message in messages[0][1]:
                input_dict = message[1]
                input_dict.update({"id": message[0]})
                if self._is_timeout(input_dict.pop("recordTime")):
                    input_list.append(input_dict)
            return input_list
        except IndexError as e:
            print("exception happened when get input", e)
            return []

    def report_response(self, message_id_list: List[str], **kwargs):
        for message_id in message_id_list:
            self.redis_client.set_response_to_list(message_id, **kwargs)
        self.redis_client.ack_message(message_id_list)

    @staticmethod
    def _is_timeout(start_time, timeout=10):
        if time.time() - float(start_time) > timeout:
            return False
        else:
            return True


if __name__ == '__main__':
    u = Utils()
    while True:
        r = u.get_input_parameters(2)
        body_list = [json.loads(i.get("body")) for i in r]
        print(body_list)
        id_list = [i.get("id") for i in r]
        time.sleep(2)
        u.report_response(id_list, rsp="test", rep2="test2")

