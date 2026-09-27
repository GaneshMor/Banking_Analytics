# Optional: deploy anywhere that runs containers (Render, Railway, Google Cloud Run, Azure, etc.)
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').environ.get('PORT','8501') + '/_stcore/health')"
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
