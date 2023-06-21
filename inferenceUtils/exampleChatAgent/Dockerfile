FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get install -y git

RUN git clone --depth 2 --branch main https://github.com/protagolabs/InferenceUtils.git
RUN pip install InferenceUtils/dist/inferenceUtils-0.0.3-py3-none-any.whl

COPY . /app/ChatAgent
WORKDIR /app/ChatAgent

RUN pip install --upgrade pip && pip  install  -r requirements.txt


ENTRYPOINT ["python","example.py"]

