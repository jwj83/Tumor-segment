ARG BASE_IMAGE=python:3.11-slim
ARG INSTALL_TORCH=0
FROM ${BASE_IMAGE}
ARG INSTALL_TORCH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COMPETITION_WORKSPACE=/2026aicompetition/workspace

WORKDIR /opt/competition
COPY requirements.txt requirements-train.txt ./
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      python -m pip install --no-cache-dir -r requirements-train.txt; \
    else \
      python -m pip install --no-cache-dir -r requirements.txt; \
    fi
COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]
