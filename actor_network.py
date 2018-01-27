import tensorflow as tf
import numpy as np


# Hyper Parameters
HIDDEN_1_SIZE = 40
HIDDEN_2_SIZE = 30

LR_A = 1e-1    # learning rate for actor
TAU = 0.01     # update rate for target network


class Actor(object):
    def __init__(self, sess, batch_size, action_dim, action_bound, s, s_, is_training):
        self.sess = sess

        self.batch_size = batch_size
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.s = s
        self.s_ = s_

        self.is_training = is_training

        self.lr = LR_A

        self.a = self._build_net(self.s)

        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Actor')
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        self.target_update = ema.apply(self.actor_vars)
        self.a_ = self._build_net(self.s_, reuse=True, getter=self.get_getter(ema))

    def get_getter(self, ema):
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var

        return ema_getter

    def _build_net(self, s, reuse=None, getter=None):
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=getter):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            hidden_1 = tf.layers.dense(s, HIDDEN_1_SIZE, activation=tf.nn.elu,
                                       kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
            hidden_2 = tf.layers.dense(hidden_1, HIDDEN_2_SIZE, activation=tf.nn.elu,
                                       kernel_initializer=init_w, bias_initializer=init_b, name='hidden_2')
            with tf.name_scope('action'):
                actions = tf.layers.dense(hidden_2, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a')
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')

        return scaled_a

    def learn(self, s, s_):   # batch update
        self.sess.run([self.train_op, self.target_update], feed_dict={self.s: s, self.s_: s_, self.is_training: True})

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={self.s: s, self.is_training: False})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.actor_vars, grad_ys=a_grads)
            for ix, grad in enumerate(self.policy_grads):
                self.policy_grads[ix] = grad / self.batch_size

        with tf.variable_scope('A_train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdagradOptimizer(-self.lr)  # (- learning rate) for ascent policy
                self.train_op = opt.apply_gradients(zip(self.policy_grads, self.actor_vars))
