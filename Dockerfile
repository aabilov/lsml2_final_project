FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app
COPY requirements.txt ./requirements.txt

COPY ./app.py ./

COPY ./model.pt ./

COPY ./return_pred.py ./

COPY ./dataset.py ./

COPY ./model.py ./

COPY templates/ ./templates/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
