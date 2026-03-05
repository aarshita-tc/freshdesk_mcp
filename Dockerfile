# Stage 1: Build
FROM python:3.12-alpine AS builder

RUN apk add --no-cache build-base

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install .

# Stage 2: Runtime (no build-base, no source code)
FROM python:3.12-alpine

WORKDIR /app

# Copy installed packages and entry point from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/freshdesk-mcp /usr/local/bin/freshdesk-mcp

# Run as non-root user
RUN adduser -D appuser
USER appuser

CMD ["freshdesk-mcp"]
