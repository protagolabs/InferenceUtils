# use ChatAgent as an example

from ChatAgent import ChatAgent
from InferenceUtils.utils_function import Utils

print("start inference")

utils = Utils()
# model_path= utils.download_and_extract("model_33069.zip")
model_path = "/tmp/tmpf_r9dgdf"
chat_agent = ChatAgent(model_path=model_path, fp16=True, parallelize=False, device="cuda:0")


def inference():
    input_raw_list = utils.get_input_parameters()
    input_list = {input.get("id"): input.get("body") for input in input_raw_list}
    model_responses = chat_agent.batch_response(input_list)

    utils.report_response(model_responses)


utils.run(inference)
