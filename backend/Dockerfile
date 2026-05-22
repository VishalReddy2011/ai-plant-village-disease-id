# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation and file copying
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first (leverages layer caching)
COPY uv.lock pyproject.toml ./
RUN uv sync --no-install-project --no-dev

# Copy the rest of the project and install it
COPY . .
RUN uv sync --frozen --no-dev


# --- Runtime Stage ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy only the virtual environment and application files from the build stage
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app /app

# Ensure the virtual environment is active by adding it to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run Uvicorn dynamically respecting the PORT environment variable injected by Cloud Run
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
