FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train.csv ./train.csv
COPY test.csv ./test.csv

COPY train.py ./train.py
COPY inference.py ./inference.py
COPY app.py ./app.py

RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]