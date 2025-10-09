FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get update --fix-missing -y
RUN apt-get install -y ffmpeg libsm6 libxrender1 libxtst6 zip p7zip-full
RUN apt-get install -y libavformat-dev libavcodec-dev libavdevice-dev \
                       libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
RUN apt-get install -y python3 python3-pip git python3-dev pkg-config htop wget

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /app/script
