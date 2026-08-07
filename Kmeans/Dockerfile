FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "user_segmentation.api:app", "--host", "0.0.0.0", "--port", "8000"]

