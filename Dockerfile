FROM python:3.12-slim

WORKDIR /app

# LightGBM wheels typically require OpenMP runtime (libgomp1) on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default sqlite location in this project is /app/db/*.sqlite3
RUN mkdir -p /app/db

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
