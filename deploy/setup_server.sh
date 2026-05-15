#!/bin/bash
# One-time setup for Ubuntu server.
# Run as: bash deploy/setup_server.sh
set -e

REPO_DIR="/home/ubuntu/recostock"
VENV_DIR="$REPO_DIR/venv"
SERVICE_NAME="intraday-bot"

echo "=== Recostock server setup ==="

# 1. Python venv
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"
echo "[OK] Python venv ready"

# 2. .env file
if [ ! -f "$REPO_DIR/deploy/.env" ]; then
    cp "$REPO_DIR/deploy/.env.example" "$REPO_DIR/deploy/.env"
    echo "[!!] Created deploy/.env — fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
fi

# 3. systemd service
sudo cp "$REPO_DIR/deploy/intraday.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "[OK] systemd service installed"

# 4. data directories
mkdir -p "$REPO_DIR/data/paper" "$REPO_DIR/data/logs" "$REPO_DIR/docs"
echo "[OK] Data directories created"

echo ""
echo "=== Next steps ==="
echo "1. Edit deploy/.env with real Telegram credentials"
echo "2. sudo systemctl start $SERVICE_NAME"
echo "3. sudo journalctl -u $SERVICE_NAME -f   # watch logs"
