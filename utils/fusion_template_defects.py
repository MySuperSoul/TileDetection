import os
import cv2
import random
import numpy as np
from util import get_root_path

if __name__ == '__main__':
    root_path = get_root_path()
    template_folder = os.path.join(
        root_path, 'data/data_guangdong/tile_round2/train_template_imgs')
    defect_folder = os.path.join(root_path,
                                 'data/data_guangdong/tile_round2/defects')

    template_names = os.listdir(template_folder)
    template_names = [
        template_names[idx]
        for idx in random.sample(range(len(template_names)), 2000)
    ]
    for name in template_names:
        template_img = cv2.imread(os.path.join(template_folder, name))
        defect_cat = random.sample([3, 4, 5, 6, 7, 8], 4)

        defect_cat = sorted(defect_cat, reverse=True)
        if defect_cat[1] == 7:
            del defect_cat[1]

        for cat in defect_cat:
            cat_folder = os.path.join(defect_folder, str(cat))
            defect_name = random.choice(os.listdir(cat_folder))
            defect_img = cv2.imread(os.path.join(cat_folder, defect_name))

            h_defect, w_defect = defect_img.shape[0], defect_img.shape[1]
            h_template, w_template = template_img.shape[0], template_img.shape[
                1]

            h_ranges = h_template - h_defect
            w_ranges = w_template - w_defect

            if h_ranges <= 0 or w_ranges <= 0:
                continue

            up_left_x = random.choice(range(w_ranges))
            up_left_y = random.choice(range(h_ranges))

            mask = 255 * np.ones(defect_img.shape, defect_img.dtype)

            template_img = cv2.seamlessClone(
                defect_img, template_img, mask,
                (int(w_defect / 2) + up_left_x, int(h_defect / 2) + up_left_y),
                cv2.NORMAL_CLONE)

            cv2.rectangle(template_img, (up_left_x, up_left_y),
                          (up_left_x + w_defect, up_left_y + h_defect),
                          (0, 0, 255))

        cv2.imwrite('fusion.jpg', template_img)
