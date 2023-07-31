FROM qctrl/ci-images:python-3.11-ci AS builder

COPY . .

RUN apt-get update && apt-get install make

RUN /scripts/install-python-dependencies.sh

WORKDIR /docs

RUN SEGMENT_WRITE_KEY=$([ -f SEGMENT_WRITE_KEY ] && cat SEGMENT_WRITE_KEY) \
    DOCSEARCH_APP_ID=$([ -f DOCSEARCH_APP_ID ] && cat DOCSEARCH_APP_ID) \
    DOCSEARCH_API_KEY=$([ -f DOCSEARCH_API_KEY ] && cat DOCSEARCH_API_KEY) \
    DOCSEARCH_INDEX_NAME=$([ -f DOCSEARCH_INDEX_NAME ] && cat DOCSEARCH_INDEX_NAME) \
    poetry run make html

FROM qctrl/ci-images:nginx-base

COPY --from=builder /docs/_build/html /usr/share/nginx/html
