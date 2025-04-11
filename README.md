# masters_project
A repo for my master's degree

# Docker


## Set-up
To set up dev environment:

```bash
docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

For prod env:

```bash
docker pull tensorflow/tensorflow:latest-gpu
```

## Running with docker

This will run a code
```bash
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:latest-gpu bash
```

This will start a `jupyter` server:
```bash
docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

## Building dockerfile
``` 
docker build -t masters_project:latest -f Dockerfile .
``` 


``` 
docker run -it --gpus all -v $(pwd):/home/app -w /home/app masters_project:latest python train.py data/mbtd/raw/Training
``` 

Or in a CPU only computer:
```
docker run -it -v $(pwd):/home/app -w /home/app masters_project:latest python train.py data/mbtd/raw/Training
```


```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \                   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


sudo apt-get install -y nvidia-container-toolkit

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```