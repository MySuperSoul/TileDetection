#!/usr/bin/env bash

rm work_dirs && rm -rf data && mkdir work_dirs && mkdir -p data/data_guangdong/tile_round2
cd data/data_guangdong/tile_round2 && mkdir train_imgs && mkdir infos

# merge the images to train_imgs
echo 'Copying images'
cp /tcdata/tile_round2_train_20210204_update/train_imgs/*.jpg train_imgs
cp /tcdata/tile_round2_train_20210208/train_imgs/*.jpg train_imgs

# merge json annotations
cd /work
echo 'Merging json annotations' && python utils/merge_train_anno_json.py
echo 'Generate train infos' && python utils/process_train_data.py

# move the pretrained models to .cache
echo 'Copying the pretrained models'
cd ~ && mkdir -p .cache/torch/hub/checkpoints
cp /external_data/resnest50_d2-7497a55b.pth ~/.cache/torch/hub/checkpoints
echo 'Done preparing data'