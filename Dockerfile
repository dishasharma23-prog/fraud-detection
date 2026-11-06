FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# hosts like Render pass the port in $PORT; 7860 when run locally
ENV PORT=7860
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
