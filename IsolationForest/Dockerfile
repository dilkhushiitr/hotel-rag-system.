FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY configs ./configs
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "fraud_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]

