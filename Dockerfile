# FROM tensorflow/tensorflow:latest-gpu
# FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
FROM base_tensorflow:latest

WORKDIR /home/app

# # Copy the pyproject.toml file
# COPY pyproject.toml .

# Copy the rest of the application
# COPY . . (can increase A LOT image size)
COPY op/ op/
COPY models/ models/
COPY experiments/ experiments/
COPY datasets/ datasets/
COPY pyproject.toml .

RUN pip install -e .
# USER base

# --data_dir tensorflow_datasets/brain_tumor_mri_dataset_kaggle/
RUN tfds build datasets/brain_tumor_mri_dataset_kaggle --overwrite

