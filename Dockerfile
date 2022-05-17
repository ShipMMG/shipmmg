FROM python:3
USER root
WORKDIR /usr/src/app
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    sudo \
    git && \
    apt-get clean

RUN pip install numpy scipy matplotlib pandas
RUN pip install pytest

RUN python -m pip install --upgrade pip && pip install \ 
    poetry \
    jupyterlab

RUN jupyter serverextension enable --py jupyterlab

ENV DEBIAN_FRONTEND dialog