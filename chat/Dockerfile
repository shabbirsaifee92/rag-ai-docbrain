FROM python:3.9-slim

WORKDIR /app

RUN pip install streamlit requests

COPY main.py .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
