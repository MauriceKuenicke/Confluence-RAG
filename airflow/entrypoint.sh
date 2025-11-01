#!/bin/bash
set -euo pipefail

AIRFLOW_HOME=${AIRFLOW_HOME:-/opt/airflow}

echo "Starting Airflow initialization..."

# Aggressively clean up ANY Airflow processes and PID files
echo "Cleaning up any stale Airflow processes and PID files..."
pkill -f "airflow" || true
pkill -f "gunicorn" || true
sleep 2

# Remove ALL stale PID files (not just some)
rm -f "${AIRFLOW_HOME}"/airflow-*.pid
rm -f "${AIRFLOW_HOME}"/airflow-webserver*.pid
rm -f "${AIRFLOW_HOME}"/airflow-scheduler.pid

# Initialize Airflow database (only once, idempotent)
echo "Initializing Airflow database..."
airflow db migrate || true

# Create admin user only if it doesn't exist (using set +e to ignore errors)
set +e
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null || true
set -e

echo "Starting Airflow components..."

# Start scheduler in background
airflow scheduler &
SCHEDULER_PID=$!

# Trap signals for proper cleanup
trap "
    echo 'Shutting down gracefully...'
    kill $SCHEDULER_PID 2>/dev/null || true
    sleep 2
    exit 0
" SIGINT SIGTERM

# Run webserver in foreground (PID 1)
exec airflow webserver --port 8080
