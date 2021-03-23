import pickle
import concurrent.futures
import os
import cv2
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class TileTestDataset(CustomDataset):
    CLASSES = (0, 1, 2, 3, 4, 5, 6)

    def _load_slides_map(self):
        self.slides_map = {
            'CAM12':
            [[720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 7200],
             [0, 720, 1440, 2160, 2880, 3600, 4320, 5040]],
            'CAM3': [[0, 720, 1440, 2160, 2880, 3196],
                     [0, 720, 1440, 2160, 2600]]
        }
        self.slides_map = {
            'CAM12': [[i, j] for i in self.slides_map['CAM12'][0]
                      for j in self.slides_map['CAM12'][1]],
            'CAM3': [[i, j] for i in self.slides_map['CAM3'][0]
                     for j in self.slides_map['CAM3'][1]]
        }

    def _crop_images(self, img_name):
        if 'CAM3' in img_name:
            slides = self.slides_map['CAM3']
        else:
            slides = self.slides_map['CAM12']

        image = cv2.imread(os.path.join(self.ann_file, img_name))
        crop_images = []
        for slide in slides:
            x, y = slide[0], slide[1]
            crop_images.append(image[y:y + self.window_size,
                                     x:x + self.window_size])
        return [img_name, crop_images]

    def load_images(self, test_img_folder, max_worker=10):
        self.all_img_names = os.listdir(test_img_folder)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.ann_file, self.all_img_names[idx]))
        results = dict(img=img)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_test_img_names(self):
        return self.all_img_names

    def get_slide_sizes(self):
        return {
            'CAM12': len(self.slides_map['CAM12']),
            'CAM3': len(self.slides_map['CAM3'])
        }

    def get_slides_positions(self, img_name):
        if 'CAM3' in img_name:
            return self.slides_map['CAM3']
        else:
            return self.slides_map['CAM12']

    def __len__(self):
        return len(self.all_img_names)


@DATASETS.register_module()
class TileTestDatasetV2(CustomDataset):
    CLASSES = (0, 1, 2, 3, 4, 5, 6)

    def load_images(self, test_img_folder, max_worker=10):
        self.all_img_names = os.listdir(test_img_folder)

    def __getitem__(self, idx):
        img_name = self.all_img_names[idx]
        image = cv2.imread(os.path.join(self.ann_file, img_name))
        center = [image.shape[1] // 2, image.shape[0] // 2]

        if 'CAM3' in img_name:
            center[1] = int(image.shape[0] * 0.55)
        length = int(image.shape[1] * 0.36)
        left_up_position = [center[0] - length, max(center[1] - length, 0)]
        image = image[max(center[1] -
                          length, 0):min(image.shape[0], center[1] + length +
                                         1),
                      center[0] - length:center[0] + length + 1, :]

        results = dict(img=image)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.all_img_names)

    def get_test_img_names(self):
        return self.all_img_names


@DATASETS.register_module()
class TileTestDatasetV3(CustomDataset):
    CLASSES = (0, 1, 2, 3, 4, 5, 6)

    def load_images(self, test_img_folder, max_worker=10):
        self.all_img_names = os.listdir(test_img_folder)

    def __getitem__(self, idx):
        img_name = self.all_img_names[idx]
        image = cv2.imread(os.path.join(self.ann_file, img_name))

        results = dict(img=image)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.all_img_names)

    def get_test_img_names(self):
        return self.all_img_names