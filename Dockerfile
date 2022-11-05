FROM python:3.7.11

ENV PYTHONUNBUFFERED 1

WORKDIR /paddle-ocr

COPY requirements.txt ./

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip3 install https://minio.dev.ftech.ai/axiom-client/axiom_client-1.8.0-py3-none-any.whl

COPY . ./

ENV PYTHONPATH "${pwd}"
ENV PATH="${PATH}:/root/.local/bin"

RUN python3 setup.py bdist_wheel

CMD python3 upload_axiom.py