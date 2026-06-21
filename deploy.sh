#!/bin/bash
set -e

# 🔴 Coloque aqui o link atual do Dropbox (ou qualquer outra fonte)
DROPBOX_URL="https://www.dropbox.com/s/xxxxxxxxxxxx/app.tar.gz?dl=1"

echo "📦 Baixando aplicação do Dropbox..."
curl -L "$DROPBOX_URL" -o /tmp/app.tar.gz

echo "📂 Extraindo arquivos em /app..."
tar xzf /tmp/app.tar.gz -C /app
rm /tmp/app.tar.gz

# Instala dependências (caso o requirements.txt tenha mudado)
if [ -f /app/requirements.txt ]; then
    echo "📚 Instalando dependências Python..."
    pip install --no-cache-dir -r /app/requirements.txt
fi

echo "🚀 Iniciando servidor..."
socat TCP6-LISTEN:8000,fork,reuseaddr TCP4:localhost:8001 &
exec gunicorn api:app -w 1 -k uvicorn.workers.UvicornWorker \
    --threads 2 \
    --bind 0.0.0.0:8001 \
    --timeout 120 \
    --keep-alive 5
