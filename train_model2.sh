#!/usr/bin/env bash

GPUS=$1

echo 'Begin training second model.'

bash ./tools/dist_train.sh configs/tile_round2/cascade_s50_rfpac_mstrain.py $GPUS --no-validate --seed=10 --deterministic

sleep 10s

echo 'Begin swa training for second model.'
bash ./tools/dist_train.sh configs/swa/swa_cascade_s50_rfp_acfpn.py $GPUS --no-validate --seed=9 --deterministic