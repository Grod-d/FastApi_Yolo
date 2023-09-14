FROM python:3.10.1-slim

WORKDIR /code

#'ffmpeg', 'libsm6', and 'libxext6' are required for OpenCV.
RUN apt-get update
RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY ./requirements.txt /code/requirements.txt
#copy pre-downloaded packages in order to miinimize network usage
#COPY ./pip_wheels/ /code/pip_wheels/

#install pre-downloaded packages in order to miinimize network usage
#RUN pip install --no-index --find-links /code/pip_wheels/ torch==1.11.0 #-r /code/requirements.txt

RUN pip install  -r /code/requirements.txt

COPY ./main.py /code/

WORKDIR /code

EXPOSE 8000

#Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]