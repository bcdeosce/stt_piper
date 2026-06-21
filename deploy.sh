#!/bin/bash
set -e

# 🔴 Substitua pela sua URL do Dropbox (agora com .zip)
DROPBOX_URL="https://www.dropbox.com/scl/fi/nfcaf9lm2q83f2ku7rxs3/app.zip?rlkey=92gmf967ddkycpqdoboj4gzzz&dl=0"

echo "📦 Baixando aplicação (formato ZIP) do Dropbox..."
curl -L "$DROPBOX_URL" -o /tmp/app.zip

echo "📂 Extraindo arquivos em /app..."
unzip -q /tmp/app.zip -d /app
rm /tmp/app.zip

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
