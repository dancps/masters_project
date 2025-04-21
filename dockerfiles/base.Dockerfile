FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /home/app

RUN pip install --upgrade tensorflow tfds-nightly apache-beam mlcroissant
