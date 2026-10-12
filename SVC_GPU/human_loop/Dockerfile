# Build from SVC_GPU: docker build -f human_loop/Dockerfile -t histoloop:latest .
FROM python:3.10.19-slim-bookworm@sha256:23f63358922e79a794f71be8f3723c84e5ccca9638af3f74456dc73d6184499e
LABEL org.opencontainers.image.title="HistoLoop" \
      org.opencontainers.image.description="Human-in-the-loop whole-slide histopathology annotation" \
      org.opencontainers.image.source="https://github.com/UFerminJamie/Cortical-and-Medullary-Segmentation"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/histoloop HISTOLOOP_HOST=0.0.0.0 HISTOLOOP_PORT=8765 \
    HISTOLOOP_ORIGINAL_MANIFEST=/data/case_split.xlsx \
    HISTOLOOP_DEVICE=cpu HISTOLOOP_ENABLE_COMPUTE=0
WORKDIR /opt/histoloop
COPY human_loop/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir pip==25.2 && pip install --no-cache-dir -r /tmp/requirements.lock
COPY __init__.py config.py data.py manifest.py outputs.py evaluation.py gpu_model.py /opt/histoloop/SVC_GPU/
COPY human_loop/ /opt/histoloop/SVC_GPU/human_loop/
RUN groupadd --gid 10001 histoloop && useradd --uid 10001 --gid histoloop --no-create-home histoloop \
    && mkdir -p /state /cache /data && chown histoloop:histoloop /state /cache
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('HISTOLOOP_PORT', '8765') + '/health', timeout=3)"
ENTRYPOINT ["/bin/bash", "/opt/histoloop/SVC_GPU/human_loop/scripts/container_entrypoint.sh"]
