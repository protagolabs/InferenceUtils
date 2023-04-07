import json
import os

import redis

host = os.getenv("REDIS_HOST", "localhost")
password = os.getenv("REDIS_PASSWORD", "")
port = os.getenv("REDIS_PORT", 6379)
stream_name = os.getenv("STREAM_NAME", "DefaultStreamName")

group_name = 'inference_group'
consumer_name = os.getenv("POD_NAME", "DefaultConsumerName")


class RedisClient:
    def __init__(self):
        pool = redis.ConnectionPool(host=host, port=port, password=password, decode_responses=True)
        self.client = redis.Redis(connection_pool=pool)
        try:
            self.client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "Group name already exists" in str(e):
                print("consumer group already exists")
                pass

    def get_data_from_stream(self, count=1):
        message = self.client.xreadgroup(group_name, "consumer_name", {stream_name: '>'}, block=0, count=count)
        if message:
            return message
        else:
            print("error: get blank message")

    def ack_message(self, message_id_list):
        for message_id in message_id_list:
            self.client.xack(stream_name, group_name, message_id)

    def set_response_to_list(self, list_name, **kwargs):
        if kwargs:
            rsp_json = json.dumps(kwargs)
            print("res_json:",rsp_json)
            self.client.rpush(list_name, rsp_json)
            return True
        else:
            print("error: response is blank")

    def get_redis_client(self):
        return self.client
