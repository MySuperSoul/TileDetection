from ..builder import PIPELINES
import mmcv
import numpy as np
import os
import cv2


@PIPELINES.register_module()
class Concat(object):
    '''
    Function: Concat two images
    '''

    def __init__(self, template_folder_path):
        super().__init__()
        self.template_folder_path = template_folder_path

    def __call__(self, results):
        if 'concat_img' not in results or results['concat_img'] is None:
            filename = results['img_info']['filename']
            template_filename = filename.split('.')[0] + '_t.jpg'
            template_img = mmcv.imread(
                os.path.join(self.template_folder_path, template_filename))
            results['img'] = np.concatenate([results['img'], template_img],
                                            axis=2)
            results['concat'] = True
        else:
            results['img'] = np.concatenate(
                [results['img'], results['concat_img']], axis=2)
            results['concat'] = True

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(self.template_folder_path)
        return repr_str


@PIPELINES.register_module()
class ConcatGrayThree(object):
    '''
    Function: Concat two gray images and their sub img
    '''

    def __init__(self, template_folder_path):
        super().__init__()
        self.template_folder_path = template_folder_path

    def __call__(self, results):
        filename = results['img_info']['filename']
        template_filename = filename.split('.')[0] + '_t.jpg'
        template_img = mmcv.imread(
            os.path.join(self.template_folder_path, template_filename))
        template_img_gray = cv2.cvtColor(template_img,
                                         cv2.COLOR_BGR2GRAY).astype(np.float32)

        origin_img_gray = cv2.cvtColor(results['img'],
                                       cv2.COLOR_BGR2GRAY).astype(np.float32)
        sub_img = origin_img_gray - template_img_gray

        results['img'] = np.dstack(
            [origin_img_gray, sub_img, template_img_gray])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(self.template_folder_path)
        return repr_str


@PIPELINES.register_module()
class SubTemplate(object):
    '''
    Function: sub img and corresponding template image
    '''

    def __init__(self, template_folder_path):
        super().__init__()
        self.template_folder_path = template_folder_path

    def __call__(self, results):
        if 'sub_img' not in results or results['sub_img'] is None:
            filename = results['img_info']['filename']
            template_filename = filename.split('.')[0] + '_t.jpg'
            template_img = mmcv.imread(
                os.path.join(self.template_folder_path, template_filename))
            results['img'] = np.clip(results['img'] - template_img, 0, 255)
            results['sub'] = True
        else:
            results['img'] = np.clip(results['img'] - results['sub_img'], 0,
                                     255)
            results['sub'] = True

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(template_path={})'.format(self.template_folder_path)
        return repr_str