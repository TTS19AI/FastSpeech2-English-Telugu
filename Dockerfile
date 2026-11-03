# Hugging Face: create a Docker Space; port 7860.
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    git espeak-ng libsndfile1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-web.txt /app/requirements-web.txt
# CPU build keeps this initial demo independent of GPU driver versions.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements-web.txt
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1 NLTK_DATA=/opt/nltk_data \
    GRADIO_ANALYTICS_ENABLED=False
RUN python -m nltk.downloader -d /opt/nltk_data cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser deploy /app/deploy
USER appuser
EXPOSE 7860
CMD ["python", "app.py"]
