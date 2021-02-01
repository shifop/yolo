#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolo import YOLOv3
from core.config import cfg
from transformers import create_optimizer
from dataset import get_data, get_dataset_by_iter
import cv2
from tqdm import tqdm
import json
from map_boxes import mean_average_precision_for_boxes
import pandas as pd
import h5py

@tf.function
def train_step(model, image_data, target, optimizers, index):
    with tf.GradientTape() as tape:
        giou_loss, conf_loss, prob_loss, total_loss = model.get_loss(image_data, target)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizers[0].apply_gradients(zip(gradients[:index], model.trainable_variables[:index]))
        optimizers[1].apply_gradients(zip(gradients[index:], model.trainable_variables[index:]))
        # optimizers.apply_gradients(zip(gradients, model.trainable_variables))
        return giou_loss, conf_loss, prob_loss, total_loss

def dev_step_draw(image_data, target, model, images):
    pred_bbox = model(image_data,False)
    pred_bbox = [pred_bbox[_*2+1] for _ in range(3)]
    pred_bbox_ = pred_bbox
    for i, image in enumerate(images):
        image_name = image.split('/')[-1].split('.')[0]
        pred_bbox = [tf.reshape(x[i], (-1, tf.shape(x)[-1])) for x in pred_bbox_]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, [1000, 1000], 608, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
        
        image = cv2.imread(image)
        image = utils.draw_bbox(image, bboxes)
        cv2.imwrite('./predict/%s_%d.jpg'%(image_name,i), image)

def dev_step(image_data, target, model, images, img_shape):
    pred_bbox = model(image_data,False)
    pred_bbox = [pred_bbox[_*2+1] for _ in range(3)]
    # pred_bbox = [target[_][0] for _ in range(3)]
    pred_bbox_ = pred_bbox
    result = {}
    for i, image in enumerate(images):
        pred_bbox = [tf.reshape(x[i], (-1, tf.shape(x)[-1])) for x in pred_bbox_]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, img_shape[i][:2], 608, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
        if image not in result:
            result[image]=[]
        result[image].extend(bboxes)
    return result

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

def load_stock_weights(model, path):
    n2v = {}
    with h5py.File(path) as f:
        for key in f:
            n2v[key] = f[key].value

    for i, x in enumerate(model.variables):
        name = x.name[:-2]
        name = name.replace('/', '|')
        if name in n2v:
            x.assign(n2v[name])
            # print("%d:%s %s Y" % (i, x.name, str(x.shape)))
        # else:
        #     print("%d:%s %s N" % (i, x.name, str(x.shape)))
    for i, x in enumerate(model.trainable_variables):
        name = x.name[:-2]
        name = name.replace('/', '|')
        if name not in n2v:
            print("%d:%s %s N" % (i, x.name, str(x.shape)))
    return model


def get_avg(model, ckpt, init_call, paths):
    print(paths)
    n2v = {}
    for path in tqdm(paths):
        ckpt.restore(path)
        model.get_loss(init_call[0], 
                ((init_call[1], init_call[4]), 
                (init_call[2], init_call[5]), 
                (init_call[3], init_call[6])))

        for v in model.variables:
            if v.name not in n2v:
                n2v[v.name] = v.numpy()
            else:
                n2v[v.name] = 0.95*n2v[v.name]+0.05*v.numpy()
    return n2v

def replace_v(model, n2v):
    for v in model.variables:
        v.assign(n2v[v.name])
    return model


if __name__=='__main__':
    # 训练、推理模式的选择，0-推理、1-训练
    tf.keras.backend.set_learning_phase(1)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_f = 'test_ep10_pretrain_aug'
    train = Dataset('train')
    i2n = ["有框表格","无框表格","页眉","图片","图表","公式","目录"]

    range_ = [320, 350]

    if not os.path.exists('./predict/%d_%d'%(range_[0],range_[1])):
        os.mkdir('./predict/%d_%d'%(range_[0],range_[1]))

    size = train.num_batchs
    train = get_dataset_by_iter(train, cfg.TRAIN.BATCH_SIZE)
    train_w, init_call = get_data(train, [608, 76, 38, 19])

    test = Dataset('test')
    dev_size = test.num_batchs

    # 读取标注数据
    raw_label = {}
    with open(cfg.TEST.ANNOT_PATH,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            p = line[0]
            tag = []
            for _ in line[1:]:
                tag.append([float(__) for __ in _.split(',')])
                tag[-1] = tag[-1][:-1]+[1]+tag[-1][-1:]
            raw_label[p] = tag

    # 获取输入图像的shape
    n2s = read_json('./data/n2s.json')

    model = YOLOv3({
        'num_class':9
    })

    # 打印variable
    model.get_loss(init_call[0], 
                ((init_call[1], init_call[4]), 
                (init_call[2], init_call[5]), 
                (init_call[3], init_call[6])))

    ckpt = tf.train.Checkpoint(model=model)

    ckpt.restore(tf.train.latest_checkpoint('./model/%s/'%(save_f)))
    # n2v = get_avg(model, ckpt, init_call, ['./model/%s/model.ckpt-%d' % (save_f, _) for _ in range(range_[0], range_[1])])
    # load_stock_weights(model,'./checkpoint/tf_model.h5')
    # ckpt.restore('./model/test_ep5/model.ckpt-400')


    images = []
    img_shape = []
    with open(cfg.TEST.ANNOT_PATH,'r',encoding='utf-8') as f:
        for line in f:
            images.append(line.strip().split(' ')[0])
            img_shape.append(n2s[images[-1]])

    global_steps = 0
    giou_losses, conf_losses, prob_losses, total_losses = [], [], [], []
    flag = True


    ###################################################################################################
    ###################################################################################################
    tqdm_d = tqdm(total=dev_size)
    dev_index = 0
    dev_batch_size = cfg.TEST.BATCH_SIZE
    result = {}
    for data_dev in test:
        image_data, target = data_dev

        result_ = dev_step(image_data, target, model, images[dev_index*dev_batch_size:(dev_index+1)*dev_batch_size], img_shape[dev_index*dev_batch_size:(dev_index+1)*dev_batch_size])
        dev_index+=1
        tqdm_d.update(1)
        
        for key in result_:
            if key not in result:
                result[key] = []
            result[key].extend([_.tolist() for _ in result_[key]])

    tqdm_d.close()

    # 将分割的图片还原
    """
    1. 加上左上点坐标还原到原图的坐标
    2. 再加一次极大值抑制
    3. 连接相邻的boxes
    """

    # 保存为csv格式
    # ImageID,LabelName,XMin,XMax,YMin,YMax
    with open('./model/%s/dev_%d_label.csv'%(save_f, global_steps),'w',encoding='utf-8') as f:
        f.write('ImageID,LabelName,XMin,YMin,XMax,YMax\n')
        for key in raw_label:
            line = raw_label[key]
            x,y = n2s[key][:2]
            # img = cv2.imread(key)
            # img = utils.draw_bbox(img, line)
            # image_name = key.split('/')[-1]
            # cv2.imwrite('./predict/%d_%d_v/%s_.jpg'%(range_[0],range_[1],image_name), img)
            x,y = n2s[key][:2]
            for _ in line:
                f.write('%s,%s,%f,%f,%f,%f\n'%(key, i2n[int(_[-1])], _[0]/y, _[1]/x, _[2]/y, _[3]/x))

    with open('./model/%s/dev_%d.csv'%(save_f, global_steps),'w',encoding='utf-8') as f:
        f.write('ImageID,LabelName,Conf,XMin,YMin,XMax,YMax\n')
        for key in result:
            line = result[key]
            img = cv2.imread(key)
            img = utils.draw_bbox(img, line)
            image_name = key.split('/')[-1].split('.')[0]
            cv2.imwrite('./predict/%d_%d/%s.jpg'%(range_[0],range_[1],image_name), img)
            x,y = n2s[key][:2]
            for _ in line:
                f.write('%s,%s,%f,%f,%f,%f,%f\n'%(key, i2n[int(_[-1])],_[-2], _[0]/y, _[1]/x, _[2]/y, _[3]/x))

    ann = pd.read_csv('./model/%s/dev_%d_label.csv'%(save_f, global_steps))
    det = pd.read_csv('./model/%s/dev_%d.csv'%(save_f, global_steps))
    ann = ann[['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].values
    det_ = det[['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det_, iou_threshold=0.5)

    det_ = det[det.Conf>=0.5][['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']].values
    mean_ap, average_precisions = mean_average_precision_for_boxes(ann, det_, iou_threshold=0.5)

