FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ADD . /code

LABEL maintainer="sr.steinkamp@mailbox.org"

# Based on tutorials
# https://blog.softwaremill.com/setting-up-tensorflow-with-gpu-acceleration-the-quick-way-add80cd5c988
# https://winsmarts.com/easiest-way-to-setup-a-tensorflow-python3-environment-with-docker-5fc3ec0f6df1

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN python -m pip install --upgrade pip setuptools wheel

COPY ./requirements.txt requirements.txt

# Install the requirements
RUN python -m pip install -r requirements.txt

# Only needed for Jupyter
EXPOSE 8888
