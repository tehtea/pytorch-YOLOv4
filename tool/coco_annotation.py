# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotation.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-json_file_path', type=str,
                        default='D:/cocoDataset/annotations_trainval2017/annotations/instances_val2017.json',
                        help='path to annotations json', dest='json_file_path')
    parser.add_argument('-images_dir_path', type=str,
                        default='D:/cocoDataset/val2017/val2017',
                        help='path to images directory that annotations refer to', dest='images_dir_path')
    parser.add_argument('-output_path', type=str,
                        default='./data/val.txt',
                        help='Output path for transformed annotations', dest='output_path')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    """hyper parameters"""
    json_file_path = args.json_file_path
    images_dir_path = args.images_dir_path
    output_path = args.output_path

    """load json file"""
    name_box_id = defaultdict(list)
    id_name = dict()
    with open(json_file_path, encoding='utf-8') as f:
        data = json.load(f)

    """generate labels"""
    images = data['images']
    annotations = data['annotations']
    for ant in tqdm(annotations):
        image_id = ant['image_id']
        chosen_image = list(filter(lambda image: image['id'] == image_id, images))
        if not len(chosen_image):
            continue
        chosen_image = chosen_image[0]
        name = os.path.join(images_dir_path, chosen_image['file_name'])
        cat = ant['category_id']

        if cat >= 1 and cat <= 11:
            cat = cat - 1
        elif cat >= 13 and cat <= 25:
            cat = cat - 2
        elif cat >= 27 and cat <= 28:
            cat = cat - 3
        elif cat >= 31 and cat <= 44:
            cat = cat - 5
        elif cat >= 46 and cat <= 65:
            cat = cat - 6
        elif cat == 67:
            cat = cat - 7
        elif cat == 70:
            cat = cat - 9
        elif cat >= 72 and cat <= 82:
            cat = cat - 10
        elif cat >= 84 and cat <= 90:
            cat = cat - 11

        name_box_id[name].append([ant['bbox'], cat])

    """write to txt"""
    with open(output_path, 'w') as f:
        for key in tqdm(name_box_id.keys()):
            f.write(key)
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = x_min + int(info[0][2])
                y_max = y_min + int(info[0][3])

                box_info = " %d,%d,%d,%d,%d" % (
                    x_min, y_min, x_max, y_max, int(info[1]))
                f.write(box_info)
            f.write('\n')
