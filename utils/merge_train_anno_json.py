import json
import os
from util import get_root_path

if __name__ == '__main__':
    root_path = get_root_path()
    train_1_json_path = '/tcdata/tile_round2_train_20210204_update/train_annos.json'

    train_2_json_path = '/tcdata/tile_round2_train_20210208/train_annos.json'
    train_json_path = '/work/data/data_guangdong/tile_round2/train_annos.json'

    with open(train_1_json_path, 'r') as f:
        annos_one = json.load(f)

    with open(train_2_json_path, 'r') as f:
        annos_two = json.load(f)

    annos = annos_one + annos_two
    with open(train_json_path, 'w') as f:
        json.dump(annos, f)