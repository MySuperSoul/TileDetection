from util import test_predictions, get_root_path
import os

if __name__ == '__main__':
    root_path = get_root_path()
    test_predictions(
        predict_json_file=os.path.join(root_path,
                                       'prediction_result/result.json'),
        image_dir=os.path.join(
            root_path,
            'data/data_guangdong/tile_round1_testB_20210128/testB_imgs'),
        result_img_save_dir=os.path.join(root_path, 'test_result'))
