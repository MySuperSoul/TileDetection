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
from ensemble_boxes import *

exp_compositions = {
    'cascade_s50_rfp': {
        'config':
        '/work/configs/tile_round2/cascade_s50_rfp_mstrain.py',
        'checkpoint':
        '/work/work_dirs/round2/swa_cascade_s50_rfp_mstrain_aug_v2/swa_model_12.pth'
    },
    'cascade_s50_rfpac': {
        'config':
        '/work/configs/tile_round2/cascade_s50_rfpac_mstrain.py',
        'checkpoint':
        '/work/work_dirs/round2/swa_cascade_s50_rfp_mstrain_acfpn/swa_model_12.pth'
    }
}
exp = exp_compositions['cascade_s50_rfp']
config_file = exp['config']
checkpoint_file = exp['checkpoint']

e_exp = exp_compositions['cascade_s50_rfpac']
e_config_file = e_exp['config']
e_checkpoint_file = e_exp['checkpoint']


class MyServer(inferServer):

    def __init__(self, model, e_model, using_pair=True):
        super().__init__(model)
        log.i('Init myserver now')

        device_1 = torch.device("cuda:0")
        device_2 = torch.device("cuda:1")

        self.device_1 = device_1
        self.device_2 = device_2

        self.using_pair = using_pair
        self.model = model.to(device_1)
        self.model.eval()

        self.e_model = e_model.to(device_2)
        self.e_model.eval()

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

        e_predictions = inference_detector(model=self.e_model, img=data)[0]

        end_time = time.time()
        print('Inference time: {} seconds'.format(end_time - start_time))

        ret = {
            'img': data,
            'predictions': predictions,
            'e_predictions': e_predictions
        }

        return ret

    def post_predictions(self, predictions, img_shape):
        bboxes_list, scores_list, labels_list = [], [], []
        for i, bboxes in enumerate(predictions):
            if len(bboxes) > 0 and i != 0:
                detect_label = i
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax, score = bbox.tolist()

                    xmin /= img_shape[1]
                    ymin /= img_shape[0]
                    xmax /= img_shape[1]
                    ymax /= img_shape[0]
                    bboxes_list.append([xmin, ymin, xmax, ymax])
                    scores_list.append(score)
                    labels_list.append(detect_label)

        return bboxes_list, scores_list, labels_list

    def ensemble(self,
                 predictions,
                 e_predictions,
                 img_shape,
                 method='weighted_boxes_fusion',
                 weights=[1.5, 1],
                 iou_thr=0.5,
                 skip_box_thr=0.0001,
                 sigma=0.1):
        bboxes, scores, labels = self.post_predictions(predictions, img_shape)
        e_bboxes, e_scores, e_labels = self.post_predictions(
            e_predictions, img_shape)
        bboxes_list = [bboxes, e_bboxes]
        scores_list = [scores, e_scores]
        labels_list = [labels, e_labels]

        bboxes, scores, labels = eval(method)(
            bboxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr)
        return bboxes, scores, labels

    def post_process(self, data):
        predictions = data['predictions']
        e_predictions = data['e_predictions']
        img_shape = data['img'].shape[:2]

        out = self.ensemble(predictions, e_predictions, img_shape)

        predict_rslt = []

        max_score = -1
        score_thr = 0.3

        for (box, score, label) in zip(*out):
            xmin, ymin, xmax, ymax = box.tolist()
            xmin, ymin, xmax, ymax = round(
                float(xmin) * img_shape[1],
                2), round(float(ymin) * img_shape[0],
                          2), round(float(xmax) * img_shape[1],
                                    2), round(float(ymax) * img_shape[0], 2)
            max_score = max(max_score, score)

            if xmax - xmin < 3 or ymax - ymin < 3:
                continue
            dict_instance = dict()
            dict_instance['name'] = self.filename
            dict_instance['category'] = int(label)
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
    e_model = init_detector(
        config=e_config_file, checkpoint=e_checkpoint_file, device='cpu')
    log.i('Init model success')
    myserver = MyServer(model=model, e_model=e_model, using_pair=False)
    # myserver.debug('test.jpg')
    myserver.run(debuge=False)
