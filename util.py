from functools import reduce

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import math as m

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                            'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                    learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                                'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                    learning_rate)

    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # interval_epoch代表余弦退火之间的最低点
                 interval_epoch=[0.05, 0.15, 0.30, 0.50],
                 verbose=0):
        super().__init__()
        self.global_step = 0
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示  
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch)-1):
            self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
        self.interval_reset.append(1-self.interval_epoch[-1])
        self.conf=[0]+[int(i*self.total_steps_for_interval) for i in self.interval_epoch]+[total_steps]


    def __call__(self, step):
        with tf.name_scope("WarmUp") as name:
            # 每到一次最低点就重新更新参数
            self.global_step  = self.global_step+1
            self.global_step_for_interval = step
            if self.global_step_for_interval >= self.conf[self.interval_index]:
                self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
                self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
                self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
                self.global_step = 0
                self.interval_index += 1

            lr = cosine_decay_with_warmup(global_step=self.global_step,
                                        learning_rate_base=self.learning_rate_base,
                                        total_steps=self.total_steps,
                                        warmup_learning_rate=self.warmup_learning_rate,
                                        warmup_steps=self.warmup_steps,
                                        hold_base_rate_steps=self.hold_base_rate_steps,
                                        min_learn_rate = self.min_learn_rate)
            return lr

def get_optimization(lr, 
                    warmup_learning_rate, 
                    min_learn_rate, 
                    num_train_steps, 
                    num_warmup_steps, 
                    hold_base_rate_steps):
    reduce_lr = WarmUp(learning_rate_base=lr, 
                    total_steps=num_train_steps, 
                    warmup_learning_rate=warmup_learning_rate, 
                    warmup_steps=num_warmup_steps, 
                    hold_base_rate_steps=hold_base_rate_steps, 
                    min_learn_rate=min_learn_rate)
    
    optimizer = keras.optimizers.Adam(learning_rate=reduce_lr)
    return optimizer