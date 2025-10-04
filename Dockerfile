FROM python:3.10-slim as base

RUN apt-get update && apt-get install -y libaio-dev wget unzip
COPY requirements/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

FROM base as app

WORKDIR /app

COPY ./ /app/

CMD ["python3", "manage.py", "start"]
