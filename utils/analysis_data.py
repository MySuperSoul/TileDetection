import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def count_categories():
    category_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    category_name_map = {
        0: '背景',
        1: '边异常',
        2: '角异常',
        3: '白色点瑕疵',
        4: '浅色块瑕疵',
        5: '深色点块瑕疵',
        6: '光圈瑕疵',
        7: '记号笔',
        8: '划伤'
    }
    for info in anno_infos:
        category_map[info['category']] += 1
    for k, v in category_map.items():
        print('category: {}, times: {}'.format(category_name_map[k], v))


def analysis_box():
    all_boxes = [info['bbox'] for info in anno_infos]
    all_boxes = [[abs(bbox[0] - bbox[2]),
                  abs(bbox[1] - bbox[3])]
                 for bbox in all_boxes]  # (width, height)
    all_boxes = np.array(all_boxes)

    box_ratios = sorted([wh[0] / wh[1] for wh in all_boxes])

    plt.figure('Ratio')
    plt.scatter(range(len(box_ratios)), box_ratios, s=[1] * len(box_ratios))
    plt.xlabel('index')
    plt.ylabel('ratio')
    plt.savefig('ratio.jpg')

    # figure all categories box width and height
    plt.figure('Draw')
    plt.scatter(all_boxes[:, 0], all_boxes[:, 1], s=[1] * len(box_ratios))
    plt.xlabel('obj_width')
    plt.ylabel('obj_height')
    plt.savefig('wh_visual.jpg')
    plt.close()


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_folder = os.path.join(root_path, 'data/data_guangdong/tile_round2')
    annos_file = os.path.join(data_folder, 'train_annos.json')
    with open(annos_file, 'r') as f:
        anno_infos = json.load(f)

    count_categories()
    analysis_box()