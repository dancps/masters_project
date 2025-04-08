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


# # Set the default command
# CMD ["python", "-c", "import masters_project; print('Package installed successfully!')"]
