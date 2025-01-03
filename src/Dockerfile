
# FROM huggingface/transformers-pytorch-cpu:latest
FROM python:3.11-slim

# don't save caches
ENV PIP_NO_CACHE_DIR=1
# avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# stream outputs to logs
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY . /app
WORKDIR /app

EXPOSE 8080

# change streamlit server config in prod
ENTRYPOINT ["streamlit","run"]

CMD ["streamlit_frontend.py", "--server.port", "8080"]