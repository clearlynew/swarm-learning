FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/nsl-kdd
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl && rm /tmp/*.whl
ENV PYTHONUNBUFFERED=1
