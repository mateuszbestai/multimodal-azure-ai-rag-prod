# Alternative approach: Use shell script to start both services
# This is simpler but less robust than supervisord

FROM python:3.11-slim as backend-build
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY backend/ .

FROM node:18-alpine as frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
ENV VITE_API_URL=""
RUN npm run build

FROM python:3.11-slim

# Install nginx
RUN apt-get update && apt-get install -y nginx wget && rm -rf /var/lib/apt/lists/*

# Copy backend
WORKDIR /app
COPY --from=backend-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-build /usr/local/bin/gunicorn /usr/local/bin/gunicorn
COPY --from=backend-build /app /app

# Copy frontend
COPY --from=frontend-build /app/dist /usr/share/nginx/html

# Copy configs
COPY docker/nginx-combined.conf /etc/nginx/nginx.conf
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create user and set permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /usr/share/nginx/html /var/log/nginx /var/lib/nginx /run

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

USER appuser
WORKDIR /app
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/api/health/live || exit 1

# Use shell script to start both services
CMD ["/app/start.sh"]