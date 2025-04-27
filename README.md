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
```bash
docker build -t base_tensorflow:latest -f dockerfiles/base.Dockerfile . 
docker build -t masters_project:latest -f Dockerfile .
``` 


```bash
docker run -it --gpus all -v $(pwd):/home/app -w /home/app masters_project:latest python train.py data/mbtd/raw/Training
``` 

Or in a CPU only computer:
```bash
docker run -it -v $(pwd):/home/app -w /home/app masters_project:latest python train.py data/mbtd/raw/Training
```


Install nvidia-container-toolkit, to be able to use GPU in docker:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \                   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list


sudo apt-get install -y nvidia-container-toolkit
```


In WSL you can also install cuda
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

# Running the file 
```bash
python experiments/armnet_experiment.py -d -e resnet_experiment
```

# Building the Dataset
Creating a new dataset

```bash
pip install pip install -q tfds-nightly apache-beam mlcroissant
```

```bash
tfds new brain_tumor_mri_dataset_kaggle
```

```bash
tfds build data/datasets/brain_tumor_mri_dataset_kaggle --data_dir data/datasets/test/
```


# Refs
- [How to disable tensorflow warnings](https://github.com/tensorflow/tensorflow/issues/54499#issuecomment-1049553976)
- [How to nvdia toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Visualizing](https://arxiv.org/abs/1311.2901)
- [[Article] Do Convnets Learn Correspondence?](https://proceedings.neurips.cc/paper_files/paper/2014/file/50f6d53bcaae4f4d70d1ecf5341f6eb4-Paper.pdf)
- [Visualization of CNN](https://github.com/bsaldivaremc2/CNN_See_output)
    - Also on [video](https://www.youtube.com/watch?v=AaAdHxtQOKQ)
- [Running docker as non-root](https://medium.com/redbubble/running-a-docker-container-as-a-non-root-user-7d2e00f8ee15)



# Dataset

```
dataset/
├── test
│   ├── glioma
│   │   ├── Te-gl_0010.jpg
│   │   ├── ...
│   │   └── Te-glTr_0009.jpg
│   ├── meningioma
│   │   ├── Te-me_0010.jpg
│   │   ├── ...
│   │   └── Te-meTr_0009.jpg
│   ├── notumor
│   │   ├── Te-no_0010.jpg
│   │   ├── ...
│   │   └── Te-noTr_0009.jpg
│   └── pituitary
│       ├── Te-pi_0010.jpg
│       ├── ...
│       └── Te-piTr_0009.jpg
└── train
    ├── glioma
    │   ├── Tr-gl_0010.jpg
    │   ├── ...
    │   └── Tr-glTr_0009.jpg
    ├── meningioma
    │   ├── Tr-me_0010.jpg
    │   ├── ...
    │   └── Tr-meTr_0009.jpg
    ├── notumor
    │   ├── Tr-no_0010.jpg
    │   ├── ...
    │   └── Tr-noTr_0009.jpg
    └── pituitary
        ├── Tr-pi_0010.jpg
        ├── ...
        └── Tr-piTr_0009.jpg
```



# Folders: 

masters_project/
├── data/
│   └── experiments/
│       ├── ...
│       └── example_experiment/
│           └── checkpoints/
│               ├── dev/
│               │   └── ...
│               └── prod/
│                   └── 10-folds-20250424-093616
│                       ├── fold-1-20250424-093616
│                       ├── fold-2-20250424-093616
│                       ├── ...
│                       └── fold-10-20250424-093616
│                           ├── logs
│                           │   ├── train
│                           │   └── validation
│                           └── model
│                               └── all_epochs
├── datasets/
│   └── brain_tumor_mri_dataset_kaggle
│       └── ...
├── dockerfiles/
├── experiments/
│   └── armnet_experiment.py
├── models/
│   └── armnet.py
├── op/
│   ├── dataset.py
│   ├── experiment.py
│   └── preprocess.py
├── playground/
├── scripts/
└── utils/
