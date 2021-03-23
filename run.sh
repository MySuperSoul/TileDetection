#!/usr/bin/env bash

# first prepare data
bash prepare_data.sh
sleep 5s

# train first model
bash train_model1.sh 4
sleep 10s

# train second model
bash train_model2.sh 4
sleep 10s

# inference with model ensemble
python inference_ensemble.py