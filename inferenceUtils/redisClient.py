import json
import os
import uuid

import redis
from redis.client import Redis
from redis.sentinel import Sentinel, MasterNotFoundError

password = os.getenv("REDIS_PASSWORD", "8HHKMUuDeYVawngZGMq6")
sentinel_password = os.getenv("SENTINEL_PASSWORD", "asdfwe90312b123412b12")
port = os.getenv("REDIS_PORT", 6379)
fetch_stream_name = os.getenv("STREAM_NAME", "DefaultStreamName")

group_name = "inference_group"
consumer_name = os.getenv("HOSTNAME", f"consumer_name_{str(uuid.uuid4())[:10]}")

class RedisClient:
    def __init__(self):
        sentinel_servers = [("44.215.65.249", 26379),
                            ("34.230.174.26", 26379),
                            ("34.230.174.26", 26379)]
        master_name = "mymaster"
        sentinel = Sentinel(sentinel_servers, password=password, encoding="utf-8", sentinel_kwargs={"password": sentinel_password})
        self.client:Redis = sentinel.master_for(master_name, encoding="utf-8",decode_responses=True)
        try:
            self.client.info()
        except MasterNotFoundError as e:
            print("redis sentinel not found , try to connect local redis")
            self.client = redis.Redis(host="localhost", port=port, password=password)
        try:
            self.client.xgroup_create(fetch_stream_name, group_name, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "Group name already exists" in str(e):
                print("consumer group already exists")
                pass
        print("redis client init finished....")

    def get_data_from_stream(self, count=1):
        message = self.client.xreadgroup(group_name, consumer_name, {fetch_stream_name: ">"}, block=0, count=count)
        if message:
            return message
        else:
            print("error: get blank message")

    def set_message_to_next_stream(self, endpoint_model_id, message_body):
        # result = re.findall(r".*?__.*?__(\d*?)__.*?Class", fetch_stream_name)
        content_list = fetch_stream_name.split("__")
        stream_name = "__".join(content_list[:2] + endpoint_model_id + content_list[3:])

        self.client.xadd(stream_name, message_body)

    def ack_message(self, message_id):
        self.client.xack(fetch_stream_name, group_name, message_id)

    def set_json_response_to_list(self, list_name, json_data: str):
        if json_data:
            self.client.rpush(list_name, json_data)
            return True
        else:
            print("error: json response is blank")

    def set_response_to_list(self, list_name, **kwargs):
        if kwargs:
            rsp_json = json.dumps(kwargs)
            print("res_json:", rsp_json)
            self.client.rpush(list_name, rsp_json)
            return True
        else:
            print("error: response is blank")

    def get_redis_client(self):
        return self.client

    def close(self):
        self.client.close()
        print("success to close redis client")

