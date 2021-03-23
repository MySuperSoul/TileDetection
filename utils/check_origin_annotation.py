from util import check_annotations, get_root_path
import os

if __name__ == '__main__':
    root = get_root_path()
    check_annotations(
        anno_json_path=os.path.join(
            root,
            'data/data_guangdong/tile_round2_train_20210204/train_annos.json'))
