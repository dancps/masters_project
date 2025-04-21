FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

RUN useradd -ms /bin/bash base

RUN pip install --upgrade tensorflow 
RUN pip install tfds-nightly apache-beam mlcroissant
