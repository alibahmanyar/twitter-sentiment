FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD [ "gunicorn", "-b", "0.0.0.0:5000", "-w", "1", "--threads", "1", "-t", "1200", "main:app" ]
#CMD ["python3", "main.py"]