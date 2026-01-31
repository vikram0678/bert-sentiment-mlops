FROM python:3.10-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# This line ensures you see the progress bars clearly
RUN pip install --no-cache-dir --progress-bar on -r requirements.txt

COPY . .