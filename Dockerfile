FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/app

# # Copy the pyproject.toml file
# COPY pyproject.toml .

# Copy the rest of the application
COPY . .

# Install build dependencies
# RUN pip install --no-cache-dir build setuptools wheel

# Install the package in editable mode
RUN pip install -e .

RUN pip install tensorflow[and-cuda]

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# # Set the default command
# CMD ["python", "-c", "import masters_project; print('Package installed successfully!')"]
