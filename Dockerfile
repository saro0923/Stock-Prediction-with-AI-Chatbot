# Dockerfile
FROM python:3.12-slim

# Metadata
LABEL maintainer="Saravanan <saravanan200423@gmail.com>"
LABEL version="1.0.0"
LABEL description="A Streamlit-based stock predictor app using Python 3.12."
LABEL org.opencontainers.image.source="https://github.com/saro0923/Stock_prediction-with-LLM-Project.git"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
