FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for FastAPI (8000) and Dash (8050)
EXPOSE 8000 8050

# We use a bash script to run all 3 python processes (Feeder, Dash, FastAPI)
# But better yet, we will use docker-compose to run them as separate services
# So this Dockerfile just defines the image, the command is overridden in docker-compose.yml
