FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN python train.py

CMD ["python", "test.py"]
# docker build -t kushrohra/kushrohra_assignment3_solution .