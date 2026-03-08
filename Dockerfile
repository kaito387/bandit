FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy numba matplotlib

COPY . .

# Default: run the main simulation
CMD ["python", "simulate.py"]
