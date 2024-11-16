FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

ENV  HF_ENDPOINT=https://hf-mirror.com

COPY ./src .
COPY ./docs .

EXPOSE 8000

CMD ["python", "test.py"]
