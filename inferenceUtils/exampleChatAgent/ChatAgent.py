import json
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    StoppingCriteria
)


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result

    return func_wrapper


@timer
class BotIdentifier:
    def __init__(self, model_path, device="cpu") -> None:
        self.config = AutoConfig.from_pretrained(
            model_path,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=self.config,
        )
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False
        )

    @torch.no_grad()
    def predict(self, text: str, out_logits=False) -> int:
        text = text.lower()
        encoded_input = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        logits = self.model(**encoded_input).logits
        if out_logits:
            return logits
        else:
            predicted_class_id = logits.argmax().item()
            return self.model.config.id2label[predicted_class_id]


class MultiEosTokenStopping(StoppingCriteria):
    def __init__(self, eos_token_ids: List[int], prefix_len: int):
        self.eos_token_ids = eos_token_ids
        self.prefix_len = prefix_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        done = []
        for ids in input_ids:
            done.append(any([stop_id in ids[self.prefix_len:] for stop_id in self.eos_token_ids]))
        return all(done)


@timer
class ChatAgent:
    def __init__(self, model_path, identifier_path=None, tokenizer_path=None, device="cuda:0", fp16=False,
                 parallelize=False, bot_name="BrainBrain") -> None:
        if tokenizer_path is None:
            tokenizer_path = model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side="left")

        if not parallelize:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            if fp16:
                self.model.half()
            self.model.to(device)
        else:
            from modeling.naive_opt_mp import NaiveMP_OPTForCausalLM
            self.model = NaiveMP_OPTForCausalLM.from_pretrained(model_path)
            if fp16:
                self.model.half()
            self.model.parallelize()

        self.speaker_1 = "Human: "  # Human
        self.speaker_2 = "Sydney:"  # Bot
        self.default_instruction = ""

        self.model.eval()
        self.bot_identifier = None
        if identifier_path:
            self.bot_identifier = BotIdentifier(identifier_path, device="cpu")

        self.stop_ids = [2, 50118]  # <\s>, "\n"

        self.bot_name = bot_name

    def observe(self, json_data: str):
        data = json.loads(json_data)
        if "history" not in data:
            return -1
        elif not isinstance(data["history"], List):
            return -1
        else:
            return data

    def template(self, data):
        history = data["history"].copy()
        instruction = self.default_instruction

        for i, text in enumerate(history):
            if i % 2 == 0:
                history[i] = self.speaker_1 + history[i].strip()
            else:
                history[i] = self.speaker_2 + history[i].strip()
        if len(history) == 0:
            input_text = instruction + self.speaker_1.strip()
        elif len(history) % 2 == 0:
            input_text = instruction + ("\n".join(history)) + "\n" + self.speaker_1.strip()
        elif len(history) % 2 == 1:
            input_text = instruction + ("\n".join(history)) + "\n" + self.speaker_2.strip()

        return input_text

    def encode(self, input_text: str, device="cuda:0") -> List[torch.Tensor]:
        input_tensor = self.tokenizer(input_text, return_tensors="pt")
        input_tensor = input_tensor.to(device)
        batch_input_ids = input_tensor["input_ids"]  # Size is (1, seq_len)
        return batch_input_ids

    def batch_encode(self, batch_input_text: List[str], device="cuda:0"):
        # 1. Must set padding=True and return_tensors='pt' if we want tokenizer return a tensor directly
        # 2. padding_side must be set to 'left' when init tokenizer
        input_tensor = self.tokenizer(batch_input_text, return_tensors="pt", padding=True)
        input_tensor = input_tensor.to(device)
        return input_tensor

    # @timer
    def generate(self, input_text, method="greedy", max_len=100, echo=False, debug=False, decode_args=None):
        eos_token_id = 50118  # "\n"
        batch_input_ids = self.encode(input_text=input_text)
        _, prefix_len = batch_input_ids.size()

        if decode_args is None:
            decode_args = {
                "no_repeat_ngram_size": 3,
                "top_p": 0.8,
                "top_k": 5
            }

        eos_stopping = MultiEosTokenStopping(eos_token_ids=self.stop_ids, prefix_len=prefix_len)
        if method == "greedy":
            batch_output = self.model.generate(
                batch_input_ids,
                max_new_tokens=max_len,
                # eos_token_id=eos_token_id,
                stopping_criteria=[eos_stopping],
                no_repeat_ngram_size=decode_args["no_repeat_ngram_size"]
            )

        elif method == "nucleus":
            batch_output = self.model.generate(
                batch_input_ids,
                max_new_tokens=max_len,
                do_sample=True,
                top_p=decode_args["top_p"],
                top_k=decode_args["top_k"],
                # eos_token_id=eos_token_id,
                stopping_criteria=[eos_stopping],
                no_repeat_ngram_size=decode_args["no_repeat_ngram_size"],
            )
        else:
            return "Not implemented decoding method"

        if debug:
            print(f"DEBUG:\ndecode_args {decode_args}")
            print(f"DEBUG:\n{self.tokenizer.decode(batch_output[0])}")

        if echo:
            return self.tokenizer.decode(batch_output[0])
        else:
            # remove <pad>
            output_ids = batch_output[0].cpu().squeeze().tolist()
            output_ids = output_ids[prefix_len:]
            output_ids = [_id for _id in output_ids if _id != self.tokenizer.pad_token_id]

            return self.tokenizer.decode(output_ids)

    def batch_generate(self,
                       batch_input_text: List[str],
                       method="greedy",
                       max_len=50,
                       input_truncate_len=500,
                       echo=False,
                       debug=False,
                       decode_args=None,
                       latency_test=False):
        """Batch generation

        Args:
            batch_input_text (List[str]): A list of multi input text.
            method (str, optional): Decoding method. Defaults to "greedy".
            max_len (int, optional): Maximum number of tokens generated. Defaults to 50.
            input_truncate_len (int, optional): Input ids is truncate to `input_truncate_len` to avoid OOM when input length is too large. Defaults to 500.
            echo (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
            decode_args (_type_, optional): _description_. Defaults to None.
        """
        # 1. batch encode
        batch_inputs = self.batch_encode(batch_input_text=batch_input_text)
        _, prefix_len = batch_inputs["input_ids"].size()
        if debug:
            print(f"batch_inputs keys: {batch_inputs.keys()}, original prefix len : {prefix_len}")

        # 2. truncate
        if prefix_len > input_truncate_len:
            batch_inputs["input_ids"] = batch_inputs["input_ids"][:, -input_truncate_len:]
            batch_inputs["attention_mask"] = batch_inputs["attention_mask"][:, -input_truncate_len:]
            print(
                f"Truncate batch_input_ids, original size: {prefix_len}, truncated size: {batch_inputs['input_ids'].size(1)}")
            prefix_len = input_truncate_len

        # 3. generate
        # 3.1. set eos_stopping dynamically depending on prefix_len
        eos_token_id = 2
        if latency_test:
            eos_token_id = None
        # 3.2 generate
        if method == "greedy":
            batch_output = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_len,
                eos_token_id=eos_token_id,
                no_repeat_ngram_size=decode_args["no_repeat_ngram_size"]
            )

        elif method == "nucleus":
            batch_output = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_len,
                do_sample=True,
                top_p=decode_args["top_p"],
                top_k=decode_args["top_k"],
                eos_token_id=eos_token_id,
                no_repeat_ngram_size=decode_args["no_repeat_ngram_size"],
            )
        else:
            return "Not implemented decoding method"

        if debug:
            print(f"batch_output size: {batch_output.size()}, new tokens: {batch_output.size(1) - prefix_len}")

        if echo:
            return self.tokenizer.batch_decode(batch_output)
        else:
            # remove prefix
            batch_output = batch_output[:, prefix_len:]
            batch_output = batch_output.cpu().tolist()
            # remove pad token id and only keep ids before eos token id
            for i, ids in enumerate(batch_output):
                filtered_ids = []
                for _id in ids:
                    if _id in self.stop_ids:
                        break
                    elif _id != self.tokenizer.pad_token_id:
                        filtered_ids.append(_id)
                batch_output[i] = filtered_ids

            # batch decode doesn't require ids padded to same length
            return self.tokenizer.batch_decode(batch_output)

    def identifier(self, data: Dict) -> int:
        return self.bot_identifier.predict(data["history"][-1])

    def is_suggestion(self, data: Dict) -> bool:
        """判断是否在请求回复建议
        """
        if len(data["history"]) % 2 == 0:
            return True
        else:
            return False

    def response(self,
                 json_data: str,
                 debug=None,
                 enable_identifier=True,
                 decode_args=None) -> str:
        # 1. load json data to dict
        data = self.observe(json_data)
        if data == -1:
            return ''

        decode_args = {
            "no_repeat_ngram_size": 4,
            "top_p": 0.6,
            "top_k": 5
        }

        # 1. template
        input_text = self.template(data)
        if debug:
            print(f"Templated input_text:\n{input_text}#")

        # 3. generate response and return 
        # Use GreedySearch to get more exact response
        response = self.generate(input_text, method="greedy", max_len=300, debug=debug, decode_args=decode_args)

        # TMP: filter "Learn more"
        response = response.split("Learn more:")[0]
        # rename
        response = response.replace("Sydney", self.bot_name)
        # TMP: replace [1], [2]
        to_replace = ["[1], [2], [3]",
                      "[1] or [2]", "[2] or [3]", "[1] or [3]",
                      "[1], [2]", "[1], [3]", "[2], [3]",
                      "[1]", "[2]", "[3]"
                      ]
        for old in to_replace:
            response = response.replace(old, "from the web")

        response = response.replace(", from the web", "")
        response = response.replace("Learn more: from the web.", "")
        response = response.replace("</s>", "")  # remove eos_token
        return response

    def batch_response(self,
                       json_data_lst: Dict[str, str],
                       debug=None,
                       decode_args=None,
                       latency_test=False):

        # 1. parser json_data_lst to json and template
        request_ids = []
        data_lst = []
        batch_input_text = []
        for req_id, json_str in json_data_lst.items():
            data = self.observe(json_str)
            if data != -1:
                data = self.pre_process(data)
                request_ids.append(req_id)
                batch_input_text.append(self.template(data))

        # 3. generate
        if decode_args is None:
            decode_args = {
                "no_repeat_ngram_size": 4,
                "top_p": 0.6,
                "top_k": 5
            }
        batch_response = self.batch_generate(batch_input_text,
                                             method="greedy",
                                             max_len=50,
                                             input_truncate_len=500,
                                             debug=debug,
                                             decode_args=decode_args,
                                             latency_test=latency_test)

        # 4. post process, add req_id and return
        results = []
        for req_id, resp in zip(request_ids, batch_response):
            resp = self.post_process(resp)
            rsp_json = json.dumps({"response": resp})
            results.append({req_id: rsp_json})

        return results

    def pre_process(self, data):
        for i, utt in enumerate(data["history"]):
            data["history"][i] = utt.replace(self.bot_name, "Sydney")
        return data

    def post_process(self, resp: str):
        # TMP: filter "Learn more"
        resp = resp.split("Learn more:")[0]
        # rename
        resp = resp.replace("Sydney", self.bot_name)
        # TMP: replace [1], [2]
        to_replace = ["[1], [2], [3]",
                      "[1] or [2]", "[2] or [3]", "[1] or [3]",
                      "[1], [2]", "[1], [3]", "[2], [3]",
                      "[1]", "[2]", "[3]"
                      ]
        for old in to_replace:
            resp = resp.replace(old, "from the web")
        resp = resp.replace(", from the web", "")
        resp = resp.replace("Learn more: from the web.", "")
        resp = resp.replace("</s>", "")  # remove eos_token
        return resp

# chat_agent = ChatAgent(model_path="./instruct2.7b", fp16=True, parallelize=False)
# chat_agent = ChatAgent(model_path="./model_step_30069", fp16=True, parallelize=False, device="cuda:0")
