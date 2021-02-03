from core.dataset import Dataset
from core.config import cfg
import tensorflow as tf

def get_dataset_by_iter(trainset, batch_size, type='train'):
    def get_dataset():
        for image_data, target in trainset:
            if len(image_data)!=batch_size and type!='dev':
                continue
            target_52, boxes_52 = target[0]
            target_26, boxes_26 = target[1]
            target_13, boxes_13 = target[2]
            yield image_data, target_52, target_26, target_13, boxes_52, boxes_26, boxes_13

    return get_dataset

def get_data(fn, input_size, repeat=1, prefetch=64, shuffle=64, seed=0, count=0, type='train'):
    dataset = tf.data.Dataset.from_generator(fn,
                                             (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             (
                                                 tf.TensorShape([None, input_size[0], input_size[0], 3]),
                                                 tf.TensorShape([None, input_size[1], input_size[1], 3, 12]),
                                                 tf.TensorShape([None, input_size[2], input_size[2], 3, 12]),
                                                 tf.TensorShape([None, input_size[3], input_size[3], 3, 12]),
                                                 tf.TensorShape([None, 150, 4]),tf.TensorShape([None, 150, 4]),tf.TensorShape([None, 150, 4])
                                             )
                                            )

    dataset = dataset.repeat(count=repeat).prefetch(prefetch)
    if type=='train':
        dataset = dataset.shuffle(shuffle, seed)
    # else:
    #     dataset = dataset.shuffle(1, seed)

    for x in dataset:
        init_call = x
        break
    return dataset, init_call