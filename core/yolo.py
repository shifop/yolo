#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 13:47:10
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from tensorflow.keras import backend as K
import math


NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES         = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH



def Mish(inputs):
    return inputs * tf.tanh(tf.nn.softplus(inputs))

class DarknetConv2D_BN_Mish(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=False):
        super(DarknetConv2D_BN_Mish, self).__init__()

        padding = 'valid' if strides==(2,2) else 'same'
        self.conv = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, use_bias=use_bias, padding=padding, kernel_regularizer='l2(5e-4)')
        self.bn = tf.keras.layers.BatchNormalization()
        

    def __call__(self, input, training):

        input = self.conv(input)
        input = self.bn(input,training)
        input = Mish(input)
        return input


class DarknetConv2D_BN_Leaky(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=False):
        super(DarknetConv2D_BN_Leaky, self).__init__()

        padding = 'valid' if strides==(2,2) else 'same'
        self.conv = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, use_bias=use_bias, padding=padding, kernel_regularizer='l2(5e-4)')
        self.bn = tf.keras.layers.BatchNormalization()
        

    def __call__(self, input, training):

        input = self.conv(input)
        input = self.bn(input,training)
        input = tf.nn.leaky_relu(input, alpha=0.1)
        return input


class resblock_body(tf.keras.Model):

    def __init__(self, num_filters, num_blocks, all_narrow=True):
        super(resblock_body, self).__init__()

        self.num_blocks = num_blocks

        self.padding = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))
        self.conv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))

        self.conv2 = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel_size = (1,1))

        self.conv3 = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel_size = (1,1))

        self.conv4 = []
        for i in range(num_blocks):
            self.conv4.append([DarknetConv2D_BN_Mish(num_filters//2, kernel_size = (1,1)), DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel_size = (3,3))])

        self.conv5 = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, kernel_size = (1,1))

        self.conv6 = DarknetConv2D_BN_Mish(num_filters, kernel_size = (1,1))

    def __call__(self, input, training=False):
        preconv1 = self.padding(input)
        preconv1 = self.conv1(preconv1, training)

        shortconv = self.conv2(preconv1, training)

        mainconv = self.conv3(preconv1, training)

        for i in range(self.num_blocks):
            y = self.conv4[i][0](mainconv, training)
            y = self.conv4[i][1](y, training)
            mainconv = tf.keras.layers.add([mainconv, y])

        postconv = self.conv5(mainconv, training)

        route = tf.keras.layers.concatenate([postconv, shortconv])

        route = self.conv6(route, training)
        return route


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


class CSPdarknet53(tf.keras.Model):

    def __init__(self):
        super(CSPdarknet53, self).__init__()

        self.conv1 = DarknetConv2D_BN_Mish(filters=32, kernel_size=(3,3))
        self.resblock_body1 = resblock_body(64, 1, False)
        self.resblock_body2 = resblock_body(128, 2)
        self.resblock_body3 = resblock_body(256, 8)
        self.resblock_body4 = resblock_body(512, 8)
        self.resblock_body5 = resblock_body(1024, 4)


    def __call__(self, input, training):
        input = self.conv1(input, training)
        input = self.resblock_body1(input, training)
        input = self.resblock_body2(input, training)
        input = self.resblock_body3(input, training)
        feat1 = input
        input = self.resblock_body4(input, training)
        feat2 = input
        input = self.resblock_body5(input, training)
        feat3 = input

        return feat1, feat2, feat3

class YOLOv4(tf.keras.Model):

    def __init__(self, config):
        super(YOLOv4, self).__init__()

        NUM_CLASS = config['num_class']
        self.NUM_CLASS = NUM_CLASS

        self.ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.STRIDES = np.array(cfg.YOLO.STRIDES)

        self.padding = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))

        self.darknet_body = CSPdarknet53()

        self.conv1 = [
            DarknetConv2D_BN_Leaky(512, (1, 1)),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(512, (1, 1))
        ]

        self.max_pooling1 = [
            tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
        ]

        self.conv2 = [
            DarknetConv2D_BN_Leaky(512, (1, 1)),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(512, (1, 1))
        ]

        self.conv_and_unsample1 = [
            DarknetConv2D_BN_Leaky(256, (1,1)), 
            tf.keras.layers.UpSampling2D(2)
        ]

        self.conv3 = DarknetConv2D_BN_Leaky(256, (1,1))

        self.conv4 = [
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            DarknetConv2D_BN_Leaky(256*2, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            DarknetConv2D_BN_Leaky(256*2, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1))
        ]

        self.conv_and_unsample2 = [
            DarknetConv2D_BN_Leaky(128, (1,1)), 
            tf.keras.layers.UpSampling2D(2)
        ]

        self.conv5 = DarknetConv2D_BN_Leaky(128, (1,1))

        self.conv6 = [
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            DarknetConv2D_BN_Leaky(128*2, (3, 3)),
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            DarknetConv2D_BN_Leaky(128*2, (3, 3)),
            DarknetConv2D_BN_Leaky(128, (1, 1))
        ]

        """
        ####################################################################################
        ###################################第三个特征图######################################
        ####################################################################################
        """
        self.conv7 = DarknetConv2D_BN_Leaky(256, (3,3))
        self.conv8 = tf.keras.layers.Conv2D(len(self.ANCHORS)*(self.NUM_CLASS+5), (1,1), padding='same')
        self.conv9 = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))
        self.conv10 = [
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            DarknetConv2D_BN_Leaky(256*2, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            DarknetConv2D_BN_Leaky(256*2, (3, 3)),
            DarknetConv2D_BN_Leaky(256, (1, 1))
        ]

        """
        ####################################################################################
        ###################################第二个特征图######################################
        ####################################################################################
        """
        self.conv11 = DarknetConv2D_BN_Leaky(512, (3,3))
        self.conv12 = tf.keras.layers.Conv2D(len(self.ANCHORS)*(self.NUM_CLASS+5), (1,1), padding='same')
        self.conv13 = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))
        self.conv14 = [
            DarknetConv2D_BN_Leaky(512, (1, 1)),
            DarknetConv2D_BN_Leaky(512*2, (3, 3)),
            DarknetConv2D_BN_Leaky(512, (1, 1)),
            DarknetConv2D_BN_Leaky(512*2, (3, 3)),
            DarknetConv2D_BN_Leaky(512, (1, 1))
        ]

        """
        ####################################################################################
        ###################################第一个特征图######################################
        ####################################################################################
        """
        self.conv15 = DarknetConv2D_BN_Leaky(1024, (3,3))
        self.conv16 = tf.keras.layers.Conv2D(len(self.ANCHORS)*(self.NUM_CLASS+5), (1,1), padding='same')

    def decode(self, conv_output, STRIDES, ANCHORS):
        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + self.NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES
        # pred_wh = (tf.clip_by_value(tf.exp(conv_raw_dwdh), 1e-9, 1e3) * ANCHORS) * STRIDES
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS) * STRIDES
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def __call__(self, input, training):
        feat1, feat2, feat3 = self.darknet_body(input, training)

        P5 = feat3
        for fn in self.conv1:
            P5 = fn(P5, training)

        max_pooling1 = []
        for fn in self.max_pooling1:
            max_pooling1.append(fn(P5))

        P5 = tf.keras.layers.concatenate(max_pooling1+[P5])
        for fn in self.conv2:
            P5 = fn(P5, training)
        P5_upsample = self.conv_and_unsample1[0](P5, training)
        P5_upsample = self.conv_and_unsample1[1](P5_upsample)

        P4 = self.conv3(feat2, training)
        P4 = tf.keras.layers.concatenate([P4, P5_upsample])
        for fn in self.conv4:
            P4 = fn(P4, training)
        P4_upsample = self.conv_and_unsample2[0](P4, training)
        P4_upsample = self.conv_and_unsample2[1](P4_upsample)

        P3 = self.conv5(feat1, training)
        P3 = tf.keras.layers.concatenate([P3, P4_upsample])
        for fn in self.conv6:
            P3 = fn(P3, training)

        """
        ####################################################################################
        ###################################第三个特征图######################################
        ####################################################################################
        """
        P3_output = self.conv7(P3, training)
        P3_output = self.conv8(P3_output)

        P3_downsample = self.padding(P3)
        P3_downsample = self.conv9(P3_downsample, training)

        P4 = tf.keras.layers.concatenate([P3_downsample, P4])
        for fn in self.conv10:
            P4 = fn(P4, training)

        """
        ####################################################################################
        ###################################第二个特征图######################################
        ####################################################################################
        """
        P4_output = self.conv11(P4, training)
        P4_output = self.conv12(P4_output)

        P4_downsample = self.padding(P4)
        P4_downsample = self.conv13(P4_downsample, training)

        P5 = tf.keras.layers.concatenate([P4_downsample, P5])
        for fn in self.conv14:
            P5 = fn(P5, training)

        """
        ####################################################################################
        ###################################第一个特征图######################################
        ####################################################################################
        """
        P5_output = self.conv15(P5, training)
        P5_output = self.conv16(P5_output)

        # 解码
        output_tensors = []
        for i, conv_tensor in enumerate([P3_output, P4_output, P5_output]):
            pred_tensor = self.decode(conv_tensor, self.STRIDES[i], self.ANCHORS[i])
            output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        return output_tensors

    def get_loss(self, input, target):
        pred_result = self.__call__(input, True)
        giou_loss=conf_loss=prob_loss=0
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        return giou_loss, conf_loss, prob_loss, total_loss

    

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = (boxes1_area + boxes2_area - inter_area) + 1e-9
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area,K.epsilon())

    # 计算中心的差距
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # 计算对角线距离
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    
    v = 4*K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1],K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    ciou = tf.where(tf.math.is_nan(ciou), tf.zeros_like(ciou), ciou)
    return ciou


def compute_loss(pred, conv, label, bboxes, i=0):

    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    # giou = tf.expand_dims(box_ciou(pred_xywh, label_xywh), axis=-1)
    giou = box_ciou(pred_xywh, label_xywh)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss