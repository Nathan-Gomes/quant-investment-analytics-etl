# Portfolio Lab API and interface.
#   docker build -t portfolio-lab .
#   docker run -p 8000:8000 portfolio-lab
FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

COPY app ./app
COPY src ./src
COPY data ./data

# Cached price history lives here; mount a volume to keep it between deploys.
ENV PORTFOLIO_LAB_SOURCE=auto HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:api --host $HOST --port $PORT"]
