FROM python:3.9.19-slim-bookworm

RUN apt -qqy update && apt -qqy install gcc python3-dev wget

ARG ROOT_DIRECTORY="/app"
ARG REPOSITORY_NAME="servier-technical-test"
ARG PACKAGE_NAME="servier"
ARG PORT=8000
ARG WORKERS=4
ARG TIMEOUT=300

ENV PIP_ROOT_USER_ACTION=ignore
ENV ROOT_DIRECTORY=$ROOT_DIRECTORY
ENV PACKAGE_NAME=$PACKAGE_NAME
ENV PORT=$PORT
ENV WORKERS=$WORKERS
ENV TIMEOUT=$TIMEOUT
ENV PYTHONPATH="${PYTHONPATH}:${ROOT_DIRECTORY}"

WORKDIR $ROOT_DIRECTORY

COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip --no-cache-dir \
  && pip install poetry --no-cache-dir \
  && poetry export --without-hashes -f requirements.txt --output requirements.txt \
  && pip install -r requirements.txt --no-cache-dir

RUN mkdir $PACKAGE_NAME

COPY conf conf
COPY $PACKAGE_NAME $PACKAGE_NAME

EXPOSE $PORT

CMD gunicorn \
    --workers $WORKERS \
    --timeout $TIMEOUT \
    --bind 0.0.0.0:$PORT \
    $PACKAGE_NAME.api.app:app
