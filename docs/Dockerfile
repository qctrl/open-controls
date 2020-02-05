FROM qctrl/python-build:3.7 AS builder

COPY . .

WORKDIR /docs

RUN apt-get update && apt-get install make

RUN /scripts/install-python-dependencies.sh

RUN poetry run make html

FROM nginx:1.17-alpine

COPY --from=builder /docs/_build/html /usr/share/nginx/html