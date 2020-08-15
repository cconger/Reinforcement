from tensorflow/tensorflow:latest-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=0

RUN apt-get update
RUN apt-get -y install ffmpeg python3-opencv

RUN pip install --pre tf-agents[reverb]
RUN pip install absl-py

ADD /src /app

CMD ["python", "/app/tf_agent_train.py"]
