FROM python:3.11-slim

# Run as a non-root user for a smaller attack surface.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# Drop privileges.
RUN chown -R appuser:appuser /app
USER appuser

# Clear port for the inference service.
EXPOSE 8000

CMD ["uvicorn", "mini_serve.server:app", "--host", "0.0.0.0", "--port", "8000"]
