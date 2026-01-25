FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache--dir
requirements.txt
CMD ["gunicorn", "--bind","0.0.0.0:10000", "app:app"]

