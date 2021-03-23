import os
import json
import cv2
from util import get_root_path


def check_slide_anno(json_path, img_dir, result_img_save_dir):
    if not os.path.exists(result_img_save_dir):
        os.mkdir(result_img_save_dir)

    with open(json_path, 'r') as f:
        context = json.load(f)

    img_names = list(set([con['name'] for con in context]))[:10]
    for index, image_name in enumerate(img_names):
        image = cv2.imread(os.path.join(img_dir, image_name))
        for con in context:
            if con['name'] == image_name:
                cv2.rectangle(image,
                              (int(con['bbox'][0]), int(con['bbox'][1])),
                              (int(con['bbox'][2]), int(con['bbox'][3])),
                              (0, 0, 255))
                cv2.putText(image, str(con['category']),
                            (int(con['bbox'][0]), int(con['bbox'][1]) - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(
            os.path.join(result_img_save_dir, 'test{}.jpg'.format(index)),
            image)


if __name__ == '__main__':
    root_path = get_root_path()
    json_path = os.path.join(
        root_path,
        'data/data_guangdong/tile_round1_train_20201231/train_canny_slide_win1650_annos.json'
    )
    check_slide_anno(
        json_path=json_path,
        img_dir=os.path.join(
            root_path,
            'data/data_guangdong/tile_round1_train_20201231/croped_slide_train_win1650'),
        result_img_save_dir=os.path.join(root_path, 'test_anno'))
