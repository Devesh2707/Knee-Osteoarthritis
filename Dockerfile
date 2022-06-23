FROM python:3.8-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

ENTRYPOINT [ "streamlit", "run" ]

CMD ["app.py"]