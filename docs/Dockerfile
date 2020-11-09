FROM qctrl/python-build:3.7 AS builder

COPY . .

RUN apt-get update && apt-get install make

RUN /scripts/install-python-dependencies.sh

WORKDIR /docs

RUN poetry run make html

FROM nginx:1.19-alpine

COPY --from=builder /docs/_build/html /usr/share/nginx/html
