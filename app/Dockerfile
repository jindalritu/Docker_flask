FROM python:3.9
WORKDIR /app
EXPOSE 8000
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . .
CMD python flask_app.py