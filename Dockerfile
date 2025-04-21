# FROM tensorflow/tensorflow:latest-gpu
# FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
FROM base_tensorflow:latest

WORKDIR /home/app

# # Copy the pyproject.toml file
# COPY pyproject.toml .

# Copy the rest of the application
# COPY . . (can increase A LOT image size)
COPY op/ .
COPY models/ .
COPY experiments/ .
COPY datasets/ .
COPY pyproject.toml .

# Install build dependencies
# RUN pip install --no-cache-dir build setuptools wheel

# Install the package in editable mode
RUN pip install -e .
# RUN pip install --upgrade tensorflow tfds-nightly apache-beam mlcroissant

RUN tfds build datasets/brain_tumor_mri_dataset_kaggle --data_dir data/datasets/test/ --overwrite

# ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
# ENV XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# # Set the default command
# CMD ["python", "-c", "import masters_project; print('Package installed successfully!')"]

