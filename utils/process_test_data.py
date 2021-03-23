from util import get_root_path, sliding_crop_canny_imgs
import os

if __name__ == '__main__':
    root_path = get_root_path()
    test_folder = os.path.join(
        root_path, 'data/data_guangdong/tile_round1_testB_20210128')
    test_img_dir = os.path.join(test_folder, 'testB_imgs')
    saved_img_dir = os.path.join(test_folder, 'croped_slide_test_win1650')
    sliding_crop_canny_imgs(
        img_dir=test_img_dir,
        save_crop_dir=saved_img_dir,
        sliding_win_xsize=1650,
        sliding_win_ysize=1650,
        overlap=200)