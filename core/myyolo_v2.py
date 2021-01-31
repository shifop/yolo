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


NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS         = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES         = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


class convolutional(tf.keras.Model):

    def __init__(self, filters_shape, downsample=False, activate=True, bn=True):
        super(convolutional, self).__init__()

        self.downsample = downsample
        self.activate = activate
        self.bn = bn
        if downsample:
            self.padding = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'
            
        self.conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))

        if self.bn:
            self.bn_ = tf.keras.layers.BatchNormalization()

    def __call__(self, input, training):
        if self.downsample:
            input = self.padding(input)
        conv = self.conv(input)
        if self.bn:
            conv = self.bn_(conv, training)
        if self.activate:
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        return conv


class residual_block(tf.keras.Model):

    def __init__(self, input_channel, filter_num1, filter_num2):
        super(residual_block, self).__init__()

        self.conv1 = convolutional(filters_shape=(1, 1, input_channel, filter_num1))
        self.conv2 = convolutional(filters_shape=(3, 3, filter_num1,   filter_num2))

    def __call__(self, input, training):
        short_cut = input
        conv = self.conv1(input, training)
        conv = self.conv2(conv, training)
        residual_output = short_cut + conv
        return residual_output


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


class darknet53(tf.keras.Model):

    def __init__(self):
        super(darknet53, self).__init__()

        self.conv1 = [convolutional((3, 3,  3,  32)), convolutional((3, 3, 32,  64), downsample=True)]
        self.residual_block1 = [residual_block(64,  32, 64)]

        self.conv2 = [convolutional((3, 3,  64, 128), downsample=True)]
        self.residual_block2 = [residual_block(128,  64, 128) for _ in range(2)]

        self.conv3 = [convolutional((3, 3, 128, 256), downsample=True)]
        self.residual_block3 = [residual_block(256, 128, 256) for _ in range(8)]

        self.conv4 = [convolutional((3, 3, 256, 512), downsample=True)]
        self.residual_block4 = [residual_block(512, 256, 512) for _ in range(8)]

        self.conv5 = [convolutional((3, 3, 512, 1024), downsample=True)]
        self.residual_block5 = [residual_block(1024, 512, 1024) for _ in range(4)]

    def __call__(self, input, training):
        for m in self.conv1:
            input = m(input, training)
        for m in self.residual_block1:
            input = m(input, training)

        for m in self.conv2:
            input = m(input, training)
        for m in self.residual_block2:
            input = m(input, training)

        for m in self.conv3:
            input = m(input, training)
        for m in self.residual_block3:
            input = m(input, training)

        route_1 = input

        for m in self.conv4:
            input = m(input, training)
        for m in self.residual_block4:
            input = m(input, training)

        route_2 = input

        for m in self.conv5:
            input = m(input, training)
        for m in self.residual_block5:
            input = m(input, training)

        return route_1, route_2, input

class YOLOv3(tf.keras.Model):

    def __init__(self, config):
        super(YOLOv3, self).__init__()

        NUM_CLASS = config['num_class']
        self.NUM_CLASS = NUM_CLASS

        self.ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.STRIDES = np.array(cfg.YOLO.STRIDES)


        self.darknet53 = darknet53()

        self.conv1 = [
            convolutional((1, 1, 1024,  512)),
            convolutional((3, 3,  512, 1024)),
            convolutional((1, 1, 1024,  512)),
            convolutional((3, 3,  512, 1024)),
            convolutional((1, 1, 1024,  512))
        ]

        self.conv_lobj_branch1 = convolutional((3, 3, 512, 1024))
        self.conv_lbbox1 = convolutional((1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        self.conv2 = convolutional((1, 1, 512, 256))

        self.conv3 = [
            convolutional((1, 1, 768, 256)),
            convolutional((3, 3, 256, 512)),
            convolutional((1, 1, 512, 256)),
            convolutional((3, 3, 256, 512)),
            convolutional((1, 1, 512, 256))
        ]

        self.conv_lobj_branch2 = convolutional((3, 3, 256, 512))
        self.conv_lbbox2 = convolutional((1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        self.conv4 = convolutional((1, 1, 256, 128))

        self.conv5 = [
            convolutional((1, 1, 384, 128)),
            convolutional((3, 3, 128, 256)),
            convolutional((1, 1, 256, 128)),
            convolutional((3, 3, 128, 256)),
            convolutional((1, 1, 256, 128))
        ]

        self.conv_lobj_branch3 = convolutional((3, 3, 128, 256))
        self.conv_lbbox3 = convolutional((1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)

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
        pred_wh = (tf.exp(tf.clip_by_value(conv_raw_dwdh,1e-9,10)) * ANCHORS) * STRIDES
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def __call__(self, input, training):
        route_1, route_2, conv = self.darknet53(input, training)

        for m in self.conv1:
            conv = m(conv, training)
        conv_lobj_branch1 = self.conv_lobj_branch1(conv, training)
        conv_lbbox = self.conv_lbbox1(conv_lobj_branch1, training)
        conv = self.conv2(conv, training)
        conv = upsample(conv)
        conv = tf.concat([conv, route_2], axis=-1)

        for m in self.conv3:
            conv = m(conv, training)
        conv_lobj_branch2 = self.conv_lobj_branch2(conv, training)
        conv_mbbox = self.conv_lbbox2(conv_lobj_branch2, training)
        conv = self.conv4(conv, training)
        conv = upsample(conv)
        conv = tf.concat([conv, route_1], axis=-1)

        for m in self.conv5:
            conv = m(conv, training)
        conv_lobj_branch3 = self.conv_lobj_branch3(conv, training)
        conv_sbbox = self.conv_lbbox3(conv_lobj_branch3, training)

        # 解码
        output_tensors = []
        for i, conv_tensor in enumerate([conv_sbbox, conv_mbbox, conv_lbbox]):
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

    def get_loss_v2(self, input, target):
        pred_result = self.__call__(input, True)
        i = 2
        conv, pred = pred_result[i*2], pred_result[i*2+1]
        loss_items = compute_loss_v2(pred, conv, *target[i], i)

        return loss_items

    

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
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


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

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
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

def compute_loss_v2(pred, conv, label, bboxes, i=0):

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

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
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

    giou_loss_ = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))

    return giou_loss_, giou_loss, respond_bbox, bbox_loss_scale, giou, pred_xywh, label_xywh