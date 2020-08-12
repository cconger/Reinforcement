from tensorflow/tensorflow:latest-gpu

RUN pip install --pre tf-agents[reverb]
RUN pip install absl-py

ADD /src /app

CMD ["python", "/app/tf_agent_train.py"]
