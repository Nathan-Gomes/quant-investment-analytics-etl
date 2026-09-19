# Strata API and interface.
#   docker build -t strata .
#   docker run -p 8000:8000 strata
FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

COPY app ./app
COPY src ./src
COPY data ./data

# Cached price history lives here; mount a volume to keep it between deploys.
COPY tools ./tools

# Fill the price cache while building, so the first visitor does not wait for a
# download and the first burst of requests does not arrive from a cold start.
# Never fails the build: if the provider refuses, the frozen dataset still works.
RUN python -m tools.warm_cache || true

ENV STRATA_SOURCE=auto HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:api --host $HOST --port $PORT"]
