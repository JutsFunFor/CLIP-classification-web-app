# CLIP-classification-web-app
#### Project for fast prototyping OpenAI CLIP image classifier, fine-tuned on custom data.
Dataset was collected as product images of random chosen brands.

 `https://en.wikipedia.org/wiki/Category:Home_appliance_brands`

## Features
- Fine-tune the CLIP model on custom datasets.
- Web interface for easy testing and interaction with the model.
- Integration with ELK stack and Tensorboard for logging and monitoring.
- Containerized deployment using Docker and Docker Compose.

## Project architecture

![Диаграмма без названия(2)](https://github.com/user-attachments/assets/7964c4f9-4e53-4ff8-af21-b19ea6f05b2d)

## 1.Clone repository

```
git clone https://github.com/JutsFunFor/CLIP-classification-web-app.git
cd CLIP-classification-web-app
```

## 2.Download dataset


Dataset located in google drive and can be downloaded by following link:

```
https://drive.google.com/file/d/1X5EgL_B0oq74XNMpTYmOBjPkQreGmm8B/view?usp=sharing
```
Unzip data and place inside `./shared` folder

Project structure 
```
me@me:~/Desktop/CLIP-classification-web-app$ tree -L 2.
.
├── docker-compose.yml
├── home_appl.png
├── LICENSE
├── logstash
│   └── pipeline
├── README.md
├── shared
│   └── final_aug_dataset
├── training
│   ├── config_docker.yaml
│   ├── dataset.py
│   ├── docker_container_run.txt
│   ├── Dockerfile
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   ├── test.py
│   ├── train.py
│   └── utils.py
└── web_app
    ├── app.py
    ├── docker_container_run.txt
    ├── Dockerfile
    └── requirements.txt

6 directories, 18 files
```
## 3.Build docker images for training and inference

```
cd  ./training
sudo docker build -t clip-training:latest .
cd ../web_app
sudo docker build -t clip-inference:latest .
cd ..
```
Run Docker compose to run training and deploy application 

```
docker compose up -d
```

This will start training process and create corresponding subfdolders inside `./shared` folder

```
me@me:~/Desktop/CLIP-classification-web-app/shared$ tree -L 2 .
.
├── final_aug_dataset
│   ├── Admiral
│   ├── Alessi
│   ├── annotations.csv
│   ├── Bedazzler
│   ├── Bertazzoni
│   ├── Bialetti
│   ├── Braun
│   ├── Breville
│   ├── Bticino
│   ├── Cuisinart
│   ├── Donvier
│   ├── Electrolux Ankarsrum Assistent
│   ├── Eureka
│   ├── Faema
│   ├── Gaggia
│   ├── Giacomini
│   ├── Gongniu
│   ├── Gorenje
│   ├── Grundig
│   ├── Haden
│   ├── Henry
│   ├── InSinkErator
│   ├── Kent RO Systems
│   ├── Lofra
│   ├── Moulinex
│   ├── Mr Coffee
│   ├── Olympic Group
│   ├── OXO
│   ├── Pars Khazar
│   ├── PeerlessPremier Appliance Company
│   ├── Pifco
│   ├── Proctor Silex
│   ├── Rancilio
│   ├── Rinnai
│   ├── Rowenta
│   ├── Russell Hobbs
│   ├── Saeco
│   ├── Sunbeam Products
│   └── Sunpentown
├── logs
│   └── training.log
├── model_registry
│   └── best_model.pth
├── registry
│   ├── dataset_metadata.json
│   ├── test_indices.pkl
│   ├── train_indices.pkl
│   └── val_indices.pkl
└── runs
    └── clip_model

44 directories, 7 files

```
 1) `logs` folder contains training.log and inference.log for further passing into ELK stack
 2) `model_registry` folder contains weights of fine-tuned model
 3) `registry` folder contains dataset indicies splitted into `train/val/test` (70/15/15)% and `dataset_metatada.json` file for dataset class distribution
 4) `runs` folder contains `tensorboard` utility log

Tensorboard can be started from `./shared` folder with command 
`tensorboard --logdir ./runs`

So after you started app with docker compose you can visit main app page

`http://localhost:8502/`

Or Kibana page for visualizing both train and inference metrics

`http://localhost:5601/app/dashboards`

## Main app

![Screenshot from 2024-09-24 00-16-57](https://github.com/user-attachments/assets/3750faa9-c2f5-4350-959a-3ddeae5c15f4)

## 4.Customize process

#### Modify training config according your available resources
```
cat ./training/config_docker.yaml
```

```
batch_size: 32
num_epochs: 5
patience: 4
num_workers: 12
weight_decay: 0.00001 
initial_lr: 0.0005
gradient_clip_value: 1.0
log_dir: "/app/runs/clip_model"
dataset_path: "/app/final_aug_dataset"
annotations_path: "/app/final_aug_dataset/annotations.csv"
model_registry_dir: "/app/model_registry"
model_weights_path: "/app/best_clip_finetuned.pth"
registry_dir: "/app/registry"
rewrite_model_weights: False
```



