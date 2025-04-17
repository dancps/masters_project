#!/bin/bash
curl -L -o data/datasets/mbtd/raw/brain-tumor-mri-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset


unzip data/datasets/mbtd/raw/brain-tumor-mri-dataset.zip -d data/datasets/mbtd/raw/

mv data/datasets/mbtd/raw/Testing data/datasets/mbtd/raw/test
mv data/datasets/mbtd/raw/Training data/datasets/mbtd/raw/train