from ..builder import PIPELINES
import numpy as np
import os
import cv2
import random


@PIPELINES.register_module()
class OnlineFusion(object):

    def __init__(self, defect_folder):
        self.defect_folder = defect_folder

    def __call__(self, results):
        filename = results['img_info']['filename']
        ori_filename = filename.split('/')[-1]
        if '_t.jpg' in ori_filename:
            defect_cat = random.sample([3, 4, 5, 6, 7, 8], 4)

            template_img = results['img'].copy()
            defect_cat = sorted(defect_cat, reverse=True)

            labels = []
            bboxes = []

            if defect_cat[1] == 7:
                del defect_cat[1]

            for cat in defect_cat:
                cat_folder = os.path.join(self.defect_folder, str(cat))
                defect_name = random.choice(os.listdir(cat_folder))
                defect_img = cv2.imread(os.path.join(cat_folder, defect_name))

                h_defect, w_defect = defect_img.shape[0], defect_img.shape[1]
                h_template, w_template = template_img.shape[
                    0], template_img.shape[1]

                h_ranges = h_template - h_defect
                w_ranges = w_template - w_defect

                if h_ranges <= 0 or w_ranges <= 0:
                    continue

                up_left_x = random.choice(range(w_ranges))
                up_left_y = random.choice(range(h_ranges))

                mask = 255 * np.ones(defect_img.shape, defect_img.dtype)

                template_img = cv2.seamlessClone(
                    defect_img, template_img, mask,
                    (int(w_defect / 2) + up_left_x,
                     int(h_defect / 2) + up_left_y), cv2.NORMAL_CLONE)

                labels.append(cat)
                bboxes.append([
                    up_left_x, up_left_y, up_left_x + w_defect,
                    up_left_y + h_defect
                ])

            bboxes = np.array(bboxes).astype(np.float32)
            labels = np.array(labels).astype(np.int64)
            results['gt_bboxes'] = bboxes
            results['gt_labels'] = labels
            results['img'] = template_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(self.defect_folder)
        return repr_str