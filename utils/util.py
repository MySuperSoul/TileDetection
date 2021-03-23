import cv2
import concurrent.futures
import os
import numpy as np
import json
import copy
import random
import pickle


def get_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sliding_crop_canny_imgs(img_dir,
                            save_crop_dir,
                            sliding_win_xsize=1650,
                            sliding_win_ysize=1650,
                            overlap=200):

    def crop(img_dir, img_name, save_crop_dir, sliding_win_xsize,
             sliding_win_ysize, overlap):
        full_path = os.path.join(img_dir, img_name)
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
            # if the canny croped is invalid, change to center crop
            center = [image.shape[1] // 2, image.shape[0] // 2]
            if 'CAM3' in img_name:
                center[1] = int(image.shape[0] * 0.55)
            length = int(image.shape[1] * 0.36)
            croped_img = image[center[1] - length:center[1] + length + 1,
                               center[0] - length:center[0] + length + 1, :]
            xmin, xmax, ymin, ymax = center[0] - length, center[
                0] + length, center[1] - length, center[1] + length

        height = ymax - ymin + 1
        width = xmax - xmin + 1
        x_slides = []
        y_slides = []
        x_slide = 0
        y_slide = 0
        while True:
            if x_slide >= width - sliding_win_xsize:
                x_slide = width - sliding_win_xsize
                x_slides.append(x_slide)
                break
            else:
                x_slides.append(x_slide)
                x_slide += (sliding_win_xsize - overlap)

        while True:
            if y_slide >= height - sliding_win_ysize:
                y_slide = height - sliding_win_ysize
                y_slides.append(y_slide)
                break
            else:
                y_slides.append(y_slide)
                y_slide += (sliding_win_ysize - overlap)

        slides_pos = [(x + xmin, y + ymin) for x in x_slides for y in y_slides]
        for slide_pos in slides_pos:
            x, y = slide_pos
            croped_img = image[y:y + sliding_win_ysize,
                               x:x + sliding_win_xsize]
            croped_img_name = '{}_{}_{}_{}_{}.jpg'.format(
                img_name.split('.')[0], x, y, x + sliding_win_xsize - 1,
                y + sliding_win_ysize - 1)
            cv2.imwrite(
                os.path.join(save_crop_dir, croped_img_name), croped_img)

        print('Finish {}'.format(img_name))

    img_names = os.listdir(img_dir)
    if not os.path.exists(save_crop_dir):
        os.mkdir(save_crop_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        for img_name in img_names:
            executor.submit(crop, img_dir, img_name, save_crop_dir,
                            sliding_win_xsize, sliding_win_ysize, overlap)


def generate_canny_slide_annotations_file(anno_file, img_dir, save_anno_folder,
                                          train_anno_json_name,
                                          bk_anno_json_name,
                                          all_anno_json_name):
    '''
    params:
        anno_file: original annotation json file path
        img_dir: the processed canny slide img patches folder
        save_anno_folder: the folder for saving the generated annotations
        train_anno_json_name: name for generated train annotations (with defects, positive samples)
        bk_anno_json_name: name for all background annotations
        all_anno_json_name: name for all the samples(all positive and negative) annotations
    '''

    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    name_anno_map = {}
    for anno in annotations:
        name = anno['name']
        if name not in name_anno_map.keys():
            name_anno_map[name] = [anno]
        else:
            name_anno_map[name].append(anno)

    all_annotations = []
    new_annotations = []
    background_annotations = []

    img_names = os.listdir(img_dir)

    for img_name in img_names:
        has_box = False
        splits = (img_name.split('.')[0]).split('_')
        origin_img_name = '{}_{}_{}_{}.jpg'.format(splits[0], splits[1],
                                                   splits[2], splits[3])
        xmin, ymin, xmax, ymax = int(splits[-4]), int(splits[-3]), int(
            splits[-2]), int(splits[-1])
        annos = name_anno_map[origin_img_name]
        image_width = xmax - xmin + 1
        image_height = ymax - ymin + 1

        for ann in annos:
            anno = copy.deepcopy(ann)

            anno['image_width'] = image_width
            anno['image_height'] = image_height
            anno['name'] = img_name
            left_up_position = [xmin, ymin]
            right_down_position = [xmax, ymax]
            box = anno['bbox']

            if box[0] >= left_up_position[0] and box[0] <= right_down_position[
                    0] and box[1] >= left_up_position[1] and box[
                        1] <= right_down_position[1] and box[
                            2] >= left_up_position[0] and box[
                                2] <= right_down_position[0] and box[
                                    3] >= left_up_position[1] and box[
                                        3] <= right_down_position[1]:
                has_box = True
                anno['bbox'][0] = round(
                    max(0, anno['bbox'][0] - left_up_position[0]), 2)  # x1
                anno['bbox'][1] = round(
                    max(0, anno['bbox'][1] - left_up_position[1]), 2)  # y1
                anno['bbox'][2] = round(
                    max(0, anno['bbox'][2] - left_up_position[0]), 2)  # x2
                anno['bbox'][3] = round(
                    max(0, anno['bbox'][3] - left_up_position[1]), 2)  # y2
                new_annotations.append(anno)

        if not has_box:
            anno = dict(
                name=img_name,
                image_height=image_height,
                image_width=image_width,
                category=0,
                bbox=[])
            background_annotations.append(anno)

    select_indices = random.sample(
        range(len(background_annotations)), len(new_annotations))
    bks = [background_annotations[idx] for idx in select_indices]
    all_annotations = bks + new_annotations
    with open(os.path.join(save_anno_folder, train_anno_json_name), 'w') as f:
        json.dump(new_annotations, f, indent=4, separators=(',', ': '))
    with open(os.path.join(save_anno_folder, bk_anno_json_name), 'w') as f:
        json.dump(background_annotations, f, indent=4, separators=(',', ': '))
    with open(os.path.join(save_anno_folder, all_anno_json_name), 'w') as f:
        json.dump(all_annotations, f, indent=4, separators=(',', ': '))


def test_predictions(predict_json_file, image_dir, result_img_save_dir):
    '''
    Descriptions:
        function to test the generated submit json file
    '''
    with open(predict_json_file, 'r') as f:
        context = json.load(f)

    image_names = os.listdir(image_dir)
    # select_indices = random.sample(range(len(image_names)), 10)
    select_indices = range(10)
    select_img_names = [image_names[i] for i in select_indices]

    if not os.path.exists(result_img_save_dir):
        os.mkdir(result_img_save_dir)

    for index, image_name in enumerate(select_img_names):
        image = cv2.imread(os.path.join(image_dir, image_name))
        for con in context:
            if con['name'] == image_name:
                cv2.rectangle(image,
                              (int(con['bbox'][0]), int(con['bbox'][1])),
                              (int(con['bbox'][2]), int(con['bbox'][3])),
                              (0, 0, 255))
                cv2.putText(
                    image,
                    str(con['category']) + '_' + str(round(con['score'], 2)),
                    (int(con['bbox'][0]), int(con['bbox'][1]) - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(
            os.path.join(result_img_save_dir, 'test{}.jpg'.format(index)),
            image)


def convert_tile_annotations_toinfos(ann_file, val_num_rate, infos_save_dir,
                                     train_info_name, val_info_name,
                                     all_info_name):
    '''
    Description:
        function to convert from the anno json to the mmdetection dataset annotation infos, 
        saved in pickle format
    '''
    with open(ann_file, 'r') as f:
        text = json.load(f)

    if not os.path.exists(infos_save_dir):
        os.mkdir(infos_save_dir)

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

    val_num = int(len(total_infos) * val_num_rate)
    val_infos_indices = random.sample(range(len(total_infos)), val_num)
    train_info_indices = list(
        set(range(len(total_infos))).difference(set(val_infos_indices)))

    train_infos = [total_infos[idx] for idx in train_info_indices]
    val_infos = [total_infos[idx] for idx in val_infos_indices]

    with open(os.path.join(infos_save_dir, train_info_name), 'wb') as f:
        pickle.dump(train_infos, f)
    with open(os.path.join(infos_save_dir, val_info_name), 'wb') as f:
        pickle.dump(val_infos, f)
    with open(os.path.join(infos_save_dir, all_info_name), 'wb') as f:
        pickle.dump(total_infos, f)

    test_recover_infos(
        val_info_file=os.path.join(infos_save_dir, val_info_name),
        train_info_file=os.path.join(infos_save_dir, train_info_name),
        all_info_file=os.path.join(infos_save_dir, all_info_name))


def test_recover_infos(val_info_file, train_info_file, all_info_file):
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


def check_annotations(anno_json_path):
    with open(anno_json_path, 'r') as f:
        context = json.load(f)

    for anno in context:
        img_name = anno['name']
        box = anno['bbox']
        height = anno['image_height']
        width = anno['image_width']
        if len(box) > 0:
            if box[0] < 0 or box[0] >= width or box[2] < 0 or box[
                    2] >= width or box[1] < 0 or box[1] >= height or box[
                        3] < 0 or box[3] >= height or box[0] >= box[2] or box[
                            1] >= box[3]:
                print(anno)