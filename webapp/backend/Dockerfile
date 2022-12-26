FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD [ "gunicorn", "-b", "0.0.0.0:5000", "main:app" ]
#CMD ["python3", "app.py"]