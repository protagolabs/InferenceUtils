# use ChatAgent as an example

from ChatAgent import ChatAgent

from inferenceUtils import Utils

# os.environ['STREAM_NAME'] = 'deploy-chat-agent-test---81__userid-12345-12345__1__FirstClass'
# 1. first download model.zip(include model.bin and config.json for transforms or model.pt/pth for torch ) from netmind
# or use huggingface xxModel.from_pretrained("xxx/xxModel") directly
utils = Utils()
model_path = utils.concurrent_download("model_33069.zip", num_proc=3)
print("model_path:", model_path)

# 2. init you model with model_path
chat_agent = ChatAgent(model_path=model_path, fp16=True, parallelize=False, device="cuda:0")


# 3. define your inference function
# it should include two function call:
#   1. utils.get_input_parameters()   -> return a list of input parameters as type: List[Dict[str, any]]
#       example: [{"id": "24f2f0ba-5a8c-4625-be37-d7bf3fc46e09", "body": '{"text": "hello"}'}]
#       some parameters can be defined like batch(default=1), is_first_model(default=true), etc.
#   2. utils.report_response(model_responses)  model_responses: List[Dict[str, any]]
#       example: [{"24f2f0ba-5a8c-4625-be37-d7bf3fc46e09":"response_body_json"}]
def inference():
    input_raw_list = utils.get_input_parameters()
    input_list = {input.get("id"): input.get("body") for input in input_raw_list}
    model_responses = chat_agent.batch_response(input_list)

    utils.report_response(model_responses, use_muti_thread=True)


# 4. calling utils.run(inference) for continuous running in loop, pass interval if you like , default 0
utils.run(inference)
