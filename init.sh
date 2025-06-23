#!/bin/bash

mkdir -p dataset
cd dataset

echo "Downloading COCO 2017 dataset and LVIS v1 validation set..."

curl -O http://images.cocodataset.org/zips/val2017.zip

curl -O http://images.cocodataset.org/zips/train2017.zip

curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

curl -O https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip



unzip val2017.zip
rm val2017.zip

unzip train2017.zip
rm train2017.zip

unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

unzip lvis_v1_val.json.zip -d annotations_lvis
rm lvis_v1_val.json.zip

mv annotations_lvis/lvis_v1_val.json annotations/lvis_v1_val.json
rm -r annotations_lvis

curl -O http://images.cocodataset.org/zips/val2017.zip

unzip val2017.zip -d val2017_lvis
rm val2017.zip

cd ../checkpoints

echo "Downloading SAM 2.1 checkpoints..."

./download_ckpts.sh

cd ..

echo "Setting up Python environment..."

python3 -m venv picosam2_env

source picosam2_env/bin/activate

python3 setup.py install

cd model_compression

python3 -m pip install -r requirements.txt