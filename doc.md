
# Labelling
Use this labelling tool: https://github.com/Cartucho/OpenLabeling
Script to convert from bbox format to TensorFlow

# TensorFlow Install & Setup
Create AWS instance with "Miniconda with Python 3" AMI image (optional, else you can install Miniconda manually)

```
sudo yum install git
git clone https://github.com/asinghani/dl-workshop.git
```

## Set up Anaconda and install libraries
```
cd dl-workshop
conda env create -f environment.yml # This may take a while
sudo yum install protobuf
sudo yum install protobuf-compiler

# MUST RUN THIS EVERY TIME TO ENABLE THE PYTHON ENVIRONMENT
conda activate deep-learning
export PYTHONPATH=$PYTHONPATH:~/models/research:~/models/research/slim
```

## Install TensorFlow Object Detection
```
git clone https://github.com/tensorflow/models
cd models/research

# Newer version has slight issues on the TX2 so this command downloads a specific version
git reset --hard 490813bdb3499290633919a9867eb0bb6d346d87

# Install a different protobuf compiler just for this part
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

./bin/protoc object_detection/protos/*.proto --python_out=.
```

## Make workspace
```
cd ~
mkdir train_workspace
cd train_workspace
mkdir train_out
```

Copy your `train.record` and `test.record` into this directory (either through an FTP client, or through SCP, or by uploading to a github repo and cloning it, or by uploading to a filesharing website)

## Create label map file
The file `label_map.pbtxt` should contain the following
```
item {
  id: 1
  name: 'Robot'
}
```

## Create Config File

See `pipeline.config` in this repo as an example

## Training
Now you are ready to train your model!

```
python ~/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=pipeline.config \
    --train_dir=train_out
```
