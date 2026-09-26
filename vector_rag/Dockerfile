FROM python:3.12-slim

WORKDIR /app

COPY vector_rag/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY vector_rag/app ./app
COPY docs /docs

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
