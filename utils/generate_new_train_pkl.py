import pickle
import os
import cv2
import numpy as np
import random
from util import get_root_path

if __name__ == '__main__':
    random.seed(10)

    root_path = get_root_path()
    infos_folder = os.path.join(root_path,
                                'data/data_guangdong/tile_round2/infos')
    train_pkl = os.path.join(infos_folder, 'all.pkl')

    with open(train_pkl, 'rb') as f:
        annos = pickle.load(f)

    train_names = []
    for anno in annos:
        train_names.append(anno['filename'])

    root_path = get_root_path()
    folder = os.path.join(root_path,
                          'data/data_guangdong/tile_round2/train_imgs')

    names = os.listdir(folder)
    normal_names = list(set(names).difference(set(train_names)))

    normal_names = [
        normal_names[idx]
        for idx in random.sample(range(len(normal_names)), 1500)
    ]

    new_annos = []
    for filename in normal_names:
        img = cv2.imread(os.path.join(folder, filename))
        item = dict()
        item['filename'] = filename
        item['width'] = img.shape[1]
        item['height'] = img.shape[0]
        item['ann'] = dict(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.array([], dtype=np.int64))
        new_annos.append(item)

    generate_annos = annos + new_annos
    with open(os.path.join(infos_folder, 'all_normal.pkl'), 'wb') as f:
        pickle.dump(generate_annos, f)