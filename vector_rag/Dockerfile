FROM python:3.12-slim

WORKDIR /app

COPY vector_rag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY vector_rag/app ./app
COPY docs /docs

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
