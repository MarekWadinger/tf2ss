# Single-stage build using uv for dependency management
FROM python:3.12-slim-bookworm

# python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$VENV_PATH/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies and uv
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libopenblas-dev coinor-libipopt-dev build-essential gfortran && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory for python environment and dependencies
WORKDIR $PYSETUP_PATH

# Copy project requirement files for dependency installation
COPY pyproject.toml README.md ./
# Copy source code
COPY tf2ss/ tf2ss/

# Install runtime dependencies into a virtual environment using uv
RUN uv venv $VENV_PATH && \
    uv sync --extra benchmark

# Set working directory for application code
WORKDIR /app

COPY . /app/

# Set entrypoint to activate the virtual environment and start a shell
CMD ["/bin/bash", "-c", "source $VENV_PATH/bin/activate && exec bash"]
