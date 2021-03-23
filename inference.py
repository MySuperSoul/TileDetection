from ai_hub import inferServer, log
import json
import torch
import mmdet
from mmdet.apis import init_detector, inference_detector
import torch.nn as nn
import numpy as np
import cv2
import time
import os

exp_compositions = {
    'model1': {
        'config':
        '/work/configs/tile_round2/cascade_s50_rfp_mstrain_augv2.py',
        'checkpoint':
        '/work/work_dirs/round2/swa_cascade_s50_rfp_mstrain_aug_v2/swa_model_12.pth'
    }
}
exp = exp_compositions['model1']
config_file = exp['config']
checkpoint_file = exp['checkpoint']


class MyServer(inferServer):

    def __init__(self, model, using_pair=True):
        super().__init__(model)
        log.i('Init myserver now')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.i('Current device: ', device)

        self.device = device
        self.using_pair = using_pair
        self.model = model.to(device)
        self.model.eval()

    def pre_process(self, request):
        #json process
        start_time = time.time()

        file = request.files['img']
        file_t = request.files['img_t']
        self.filename = file.filename
        file_data = file.read()

        img = cv2.imdecode(
            np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)

        end_time = time.time()
        print('Preprocess time: {} seconds'.format(end_time - start_time))

        return img

    def pridect(self, data):
        # remember here to fetch the first one
        start_time = time.time()

        predictions = inference_detector(model=self.model, img=data)[0]

        end_time = time.time()
        print('Inference time: {} seconds'.format(end_time - start_time))

        return predictions

    def post_process(self, data):
        predictions = data
        predict_rslt = []

        max_score = -1
        score_thr = 0.28

        for i, bboxes in enumerate(predictions):
            if len(bboxes) > 0 and i != 0:
                detect_label = i
                image_name = self.filename
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax, score = bbox.tolist()
                    xmin, ymin, xmax, ymax = round(float(xmin), 2), round(
                        float(ymin), 2), round(float(xmax),
                                               2), round(float(ymax), 2)
                    max_score = max(max_score, score)

                    if xmax - xmin < 3 or ymax - ymin < 3:
                        continue

                    dict_instance = dict()
                    dict_instance['name'] = image_name
                    dict_instance['category'] = detect_label
                    dict_instance['score'] = round(float(score), 6)
                    dict_instance['bbox'] = [xmin, ymin, xmax, ymax]
                    predict_rslt.append(dict_instance)

        if max_score < score_thr:
            predict_rslt = []

        return predict_rslt

    def debug(self, filename):
        start_time = time.time()
        self.filename = filename
        img = cv2.imread(filename)

        end_time = time.time()
        print('Preprocess time: {} seconds'.format(end_time - start_time))
        data = self.pridect(img)
        data = self.post_process(data)
        return data


if __name__ == '__main__':
    model = init_detector(
        config=config_file, checkpoint=checkpoint_file, device='cpu')
    log.i('Init model success')
    myserver = MyServer(model=model, using_pair=False)
    myserver.run(debuge=False)
