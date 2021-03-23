#!/usr/bin/env bash

GPUS=$1

echo 'Begin training first model.'

bash ./tools/dist_train.sh configs/tile_round2/cascade_s50_rfp_mstrain_augv2.py $GPUS --no-validate --seed=10 --deterministic

sleep 10s

echo 'Begin swa training for first model.'
bash ./tools/dist_train.sh configs/swa/swa_cascade_s50_rfp_mstrain_augv2.py $GPUS --no-validate --seed=9 --deterministic