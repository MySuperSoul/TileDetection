import numpy as np
import json
import os
import random
import pickle
import cv2
import concurrent.futures


def convert_tile_annotations(ann_file, val_num):
    with open(ann_file, 'r') as f:
        text = json.load(f)

    total_annotations = {}
    total_infos = []
    for annotation in text:
        if annotation['name'] not in total_annotations.keys():
            total_annotations[annotation['name']] = [
                annotation['image_height'], annotation['image_width'],
                [annotation['category']], [annotation['bbox']]
            ]
        else:
            total_annotations[annotation['name']][2].append(
                annotation['category'])
            total_annotations[annotation['name']][3].append(annotation['bbox'])

    for file_name in total_annotations.keys():
        file_info = total_annotations[file_name]
        bboxes = file_info[-1]
        labels = file_info[-2]
        if len(bboxes[0]) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        else:
            bboxes = np.array(bboxes).astype(np.float32)
            labels = np.array(labels).astype(np.int64)
        total_infos.append(
            dict(
                filename=file_name,
                height=file_info[0],
                width=file_info[1],
                ann=dict(bboxes=bboxes, labels=labels)))

    val_infos_indices = random.sample(range(len(total_infos)), val_num)
    train_info_indices = list(
        set(range(len(total_infos))).difference(set(val_infos_indices)))

    train_infos = [total_infos[idx] for idx in train_info_indices]
    val_infos = [total_infos[idx] for idx in val_infos_indices]

    with open(train_info_file, 'wb') as f:
        pickle.dump(train_infos, f)
    with open(val_info_file, 'wb') as f:
        pickle.dump(val_infos, f)
    with open(all_info_file, 'wb') as f:
        pickle.dump(total_infos, f)


def test_recover_infos():
    with open(val_info_file, 'rb') as f:
        val_infos = pickle.load(f)
    with open(train_info_file, 'rb') as f:
        train_infos = pickle.load(f)
    with open(all_info_file, 'rb') as f:
        all_infos = pickle.load(f)
    assert (type(val_infos[0]) == dict)
    print(val_infos[0], len(val_infos))
    print(train_infos[0], len(train_infos))
    print(len(all_infos))


def crop_image():
    index = 0
    for img_name in os.listdir(folder):
        full_path = os.path.join(folder, img_name)
        image = cv2.imread(full_path)
        center = [image.shape[1] // 2, image.shape[0] // 2]
        if 'CAM3' in img_name:
            center[1] = int(image.shape[0] * 0.55)
        length = int(image.shape[1] * 0.36)
        crop_image = image[center[1] - length:center[1] + length + 1,
                           center[0] - length:center[0] + length + 1, :]
        cv2.imwrite(os.path.join(crop_folder, img_name), crop_image)
        print('finish image processing : {}, index: {}'.format(
            img_name, index))
        index += 1


def crop_image_canny(folder, store_folder='/home/huangyifei/croped'):

    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    def crop(img_name):
        full_path = os.path.join(folder, img_name)
        image = cv2.imread(full_path)
        resized_image = cv2.resize(
            image, (int(0.1 * image.shape[1]), int(0.1 * image.shape[0])))
        imgray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        ret, bin_pic = cv2.threshold(imgray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        median = cv2.medianBlur(bin_pic, 5)
        cannyPic = cv2.Canny(median, 10, 200)

        contours, hierarchy = cv2.findContours(cannyPic, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

        maxArea = 0
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > cv2.contourArea(
                    contours[maxArea]):
                maxArea = i

        boader = contours[maxArea]
        xmin, xmax, ymin, ymax = np.min(boader[:, :, 0]), np.max(
            boader[:, :, 0]), np.min(boader[:, :, 1]), np.max(boader[:, :, 1])

        xmin = xmin * 10 - 150
        xmax = xmax * 10 + 150
        ymin = ymin * 10 - 150
        ymax = ymax * 10 + 150

        if xmax - xmin > 1500 and ymax - ymin > 1500 and xmin >= 0 and xmax <= image.shape[
                1] and ymin >= 0 and ymax <= image.shape[0]:
            croped_img = image[ymin:ymax + 1, xmin:xmax + 1]
        else:
            center = [image.shape[1] // 2, image.shape[0] // 2]
            if 'CAM3' in img_name:
                center[1] = int(image.shape[0] * 0.55)
            length = int(image.shape[1] * 0.36)
            croped_img = image[center[1] - length:center[1] + length + 1,
                               center[0] - length:center[0] + length + 1, :]
            xmin, xmax, ymin, ymax = center[0] - length, center[
                0] + length, center[1] - length, center[1] + length

        croped_img_filename = img_name.split(
            '.')[0] + '_{}_{}_{}_{}.jpg'.format(xmin, ymin, xmax, ymax)

        cv2.imwrite(
            os.path.join(store_folder, croped_img_filename), croped_img)
        print('Finish img: {}'.format(croped_img_filename))

    img_names = os.listdir(folder)
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        for img_name in img_names:
            future = executor.submit(crop, img_name)


def update_annotation_file():
    anno_folder = '/private/huangyifei/competition/data_guangdong_cizhuan/tile_round1_train_20201231'
    anno_file = os.path.join(anno_folder, 'train_annos.json')
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    new_annotations = []
    for i in range(len(annotations)):
        img_name = annotations[i]['name']
        center = [
            annotations[i]['image_width'] // 2,
            annotations[i]['image_height'] // 2
        ]
        if 'CAM3' in img_name:
            center[1] = int(annotations[i]['image_height'] * 0.55)
        length = int(annotations[i]['image_width'] * 0.36)

        annotations[i]['image_width'] = annotations[i][
            'image_height'] = 2 * length + 1

        left_up_position = [center[0] - length, center[1] - length]
        right_down_position = [center[0] + length, center[1] + length]
        box = annotations[i]['bbox']

        if box[0] >= left_up_position[0] and box[0] <= right_down_position[
                0] and box[1] >= left_up_position[1] and box[
                    1] <= right_down_position[1] and box[2] >= left_up_position[
                        0] and box[2] <= right_down_position[0] and box[
                            3] >= left_up_position[1] and box[
                                3] <= right_down_position[1]:
            annotations[i]['bbox'][0] = round(
                max(0, annotations[i]['bbox'][0] - left_up_position[0]),
                2)  # x1
            annotations[i]['bbox'][1] = round(
                max(0, annotations[i]['bbox'][1] - left_up_position[1]),
                2)  # y1
            annotations[i]['bbox'][2] = round(
                max(0, annotations[i]['bbox'][2] - left_up_position[0]),
                2)  # x2
            annotations[i]['bbox'][3] = round(
                max(0, annotations[i]['bbox'][3] - left_up_position[1]),
                2)  # y2

            new_annotations.append(annotations[i])
        else:
            print(img_name, annotations[i]['category'])

    with open(os.path.join(anno_folder, 'train_annos_crop.json'), 'w') as f:
        json.dump(new_annotations, f, indent=4, separators=(',', ': '))


def update_canny_annotation_file():
    anno_folder = '/private/huangyifei/competition/data_guangdong_cizhuan/tile_round1_train_20201231'
    anno_file = os.path.join(anno_folder, 'train_annos.json')
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    new_annotations = []

    canny_img_folder = '/ssd/huangyifei/data_guangdong/tile_round1_train_20201231/croped_train'
    img_names = os.listdir(canny_img_folder)

    infos = {}
    for img_name in img_names:
        splits = (img_name.split('.')[0]).split('_')
        origin_img_name = '{}_{}_{}_{}.jpg'.format(splits[0], splits[1],
                                                   splits[2], splits[3])
        xmin, ymin, xmax, ymax = int(splits[-4]), int(splits[-3]), int(
            splits[-2]), int(splits[-1])
        infos[origin_img_name] = [xmin, ymin, xmax, ymax, img_name]

    for anno in annotations:
        xmin, ymin, xmax, ymax, image_name = infos[anno['name']]
        image_width = xmax - xmin + 1
        image_height = ymax - ymin + 1

        anno['image_width'] = image_width
        anno['image_height'] = image_height
        anno['name'] = image_name

        left_up_position = [xmin, ymin]
        right_down_position = [xmax, ymax]
        box = anno['bbox']

        if box[0] >= left_up_position[0] and box[0] <= right_down_position[
                0] and box[1] >= left_up_position[1] and box[
                    1] <= right_down_position[1] and box[2] >= left_up_position[
                        0] and box[2] <= right_down_position[0] and box[
                            3] >= left_up_position[1] and box[
                                3] <= right_down_position[1]:
            anno['bbox'][0] = round(
                max(0, anno['bbox'][0] - left_up_position[0]), 2)  # x1
            anno['bbox'][1] = round(
                max(0, anno['bbox'][1] - left_up_position[1]), 2)  # y1
            anno['bbox'][2] = round(
                max(0, anno['bbox'][2] - left_up_position[0]), 2)  # x2
            anno['bbox'][3] = round(
                max(0, anno['bbox'][3] - left_up_position[1]), 2)  # y2

            new_annotations.append(anno)

    with open(
            os.path.join(
                '/ssd/huangyifei/data_guangdong/tile_round1_train_20201231/annotations',
                'train_annos_canny.json'), 'w') as f:
        json.dump(new_annotations, f, indent=4, separators=(',', ': '))


def check_annotation(ann_file):
    with open(ann_file, 'r') as f:
        annotations = json.load(f)

    for anno in annotations:
        img_name = anno['name']
        splits = (img_name.split('.')[0]).split('_')
        xmin, ymin, xmax, ymax = int(splits[-4]), int(splits[-3]), int(
            splits[-2]), int(splits[-1])
        box = anno['bbox']
        height = anno['image_height']
        width = anno['image_width']
        if len(box) > 0:
            if box[0] < 0 or box[0] >= width or box[2] < 0 or box[
                    2] >= width or box[1] < 0 or box[1] >= height or box[
                        3] < 0 or box[3] >= height or box[0] >= box[2] or box[
                            1] >= box[3]:
                print(anno)

        if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
            print(anno)


def check_img_dir(img_dir):
    img_names = os.listdir(img_dir)

    for img_name in img_names:
        splits = (img_name.split('.')[0]).split('_')
        xmin, ymin, xmax, ymax = int(splits[-4]), int(splits[-3]), int(
            splits[-2]), int(splits[-1])

        if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0 or (
                xmax - xmin < 1000) or (ymax - ymin < 1000):
            print(img_name)


if __name__ == '__main__':
    # some connstant information for converter
    data_folder = '/ssd/huangyifei/data_guangdong/tile_round1_train_20201231'
    folder = '/private/huangyifei/competition/data_guangdong_cizhuan/tile_round1_train_20201231/train_imgs/'
    test_folder = '/private/huangyifei/competition/data_guangdong_cizhuan/tile_round1_testA_20201231/testA_imgs'
    crop_folder = '/private/huangyifei/competition/data_guangdong_cizhuan/tile_round1_train_20201231/crop_train_imgs/'
    ann_file = os.path.join(data_folder, 'annotations/train_annos_canny.json')
    train_info_file = os.path.join(data_folder,
                                   'infos/train_canny_win1600.pkl')
    val_info_file = os.path.join(data_folder, 'infos/val_canny_win1600.pkl')
    all_info_file = os.path.join(data_folder, 'infos/all_canny_win1600.pkl')
    val_sample_nums = 100

    # update_canny_annotation_file()
    convert_tile_annotations(
        ann_file=os.path.join(
            data_folder,
            'annotations/train_annos_canny_slide_all_win1600.json'),
        val_num=1000)
    test_recover_infos()
    check_annotation(
        ann_file=os.path.join(
            data_folder,
            'annotations/train_annos_canny_slide_all_win1600.json'))
    # update_annotation_file()
    # crop_image_canny(
    #     folder,
    #     '/ssd/huangyifei/data_guangdong/tile_round1_train_20201231/croped_train'
    # )
    # crop_image_canny(
    #     test_folder,
    #     '/ssd/huangyifei/data_guangdong/tile_round1_testA_20201231/croped_test'
    # )
    # check_img_dir(
    #     '/ssd/huangyifei/data_guangdong/tile_round1_train_20201231/croped_slide_train_win2k'
    # )
