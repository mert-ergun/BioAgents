# Build explicitly from the repository root:
# docker build -f docker/coder.Dockerfile -t bioagents-sandbox:local .
# Coder/ML/DL share this image; Boltz and PyTDC use independent Python venvs.
ARG PYTHON_IMAGE=python:3.12.11-bullseye
FROM ${PYTHON_IMAGE} AS environments

# CPU is the portable default. Override both arguments for a CUDA wheel/index.
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    BIOAGENTS_VENV_ROOT=/opt/venvs \
    BIOAGENTS_INSTALL_CACHE=/opt/bioagents/install-cache \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venvs/coder/bin:$PATH \
    VIRTUAL_ENV=/opt/venvs/coder \
    HOME=/tmp/sandbox-home \
    JUPYTER_RUNTIME_DIR=/tmp/sandbox-home/jupyter/runtime \
    JUPYTER_DATA_DIR=/tmp/sandbox-home/jupyter/data \
    XDG_CACHE_HOME=/tmp/sandbox-home/cache \
    MPLCONFIGDIR=/tmp/sandbox-home/matplotlib
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/sandbox-home/jupyter/runtime /tmp/sandbox-home/jupyter/data \
        /tmp/sandbox-home/cache /tmp/sandbox-home/matplotlib \
    && chmod -R 777 /tmp/sandbox-home

# PyTDC's cellxgene-census pin supports Python <3.12.
RUN /usr/local/bin/python -m pip install --no-cache-dir uv==0.12.5 \
    && uv python install --no-cache --no-bin 3.11

COPY bioagents/sandbox/install_environments.py /opt/bioagents/install_environments.py
COPY bioagents/sandbox/requirements/coder*.txt /opt/bioagents/requirements/
RUN TORCH_VERSION="${TORCH_VERSION}" TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
    /usr/local/bin/python /opt/bioagents/install_environments.py coder
COPY bioagents/sandbox/requirements/boltz*.txt /opt/bioagents/requirements/
RUN TORCH_VERSION="${TORCH_VERSION}" TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
    /usr/local/bin/python /opt/bioagents/install_environments.py boltz
COPY bioagents/sandbox/requirements/pytdc*.txt /opt/bioagents/requirements/
RUN TORCH_VERSION="${TORCH_VERSION}" TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
    /usr/local/bin/python /opt/bioagents/install_environments.py pytdc

# Reuse the completed environments without copying the CUDA stack again.
FROM environments
RUN rm -rf /opt/bioagents/install-cache
ENV BIOAGENTS_INSTALL_CACHE=
COPY bioagents/sandbox/python_environments.py /usr/local/bin/bioagents-python
RUN chmod +x /usr/local/bin/bioagents-python && command -v obabel

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin=*"]
