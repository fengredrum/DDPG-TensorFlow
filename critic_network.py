import tensorflow as tf


# Hyper Parameters
HIDDEN_1_SIZE = 40
HIDDEN_2_SIZE = 30

LR_C = 1e-1    # learning rate for critic
TAU = 0.01     # update rate for target network
GAMMA = 0.95    # reward discount


class Critic(object):
    def __init__(self, sess, batch_size, state_dim, action_dim, a, a_, s, r, s_, is_training):
        self.sess = sess

        self.batch_size = batch_size
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.s = s
        self.r = r
        self.s_ = s_

        self.a = a
        self.a_ = a_

        self.is_training = is_training

        self.lr = LR_C
        self.gamma = GAMMA

        self.q = self._build_net(self.s, self.a)

        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        self.target_update = ema.apply(self.critic_vars)
        self.q_ = self._build_net(self.s_, self.a_, reuse=True, getter=self.get_getter(ema))

        with tf.variable_scope('target_q'):
            self.target_q = self.r + self.gamma * self.q_
        with tf.variable_scope('TD_error'):
            self.loss = tf.squared_difference(self.target_q, self.q)

        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.loss, xs=self.critic_vars)
            for ix, grad in enumerate(self.policy_grads):
                self.policy_grads[ix] = grad / self.batch_size
        with tf.variable_scope('C_train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdagradOptimizer(self.lr)
                self.train_op = opt.apply_gradients(zip(self.policy_grads, self.critic_vars))

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]

    def get_getter(self, ema):
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var

        return ema_getter

    def _build_net(self, s, a, reuse=None, getter=None):
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=getter):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            hidden_1 = tf.layers.dense(s, HIDDEN_1_SIZE, activation=tf.nn.elu,
                                       kernel_initializer=init_w, bias_initializer=init_b, name='hidden_1')
            with tf.variable_scope('hidden_2'):
                w2_s = tf.get_variable('w2_s', [HIDDEN_1_SIZE, HIDDEN_2_SIZE], initializer=init_w)
                w2_a = tf.get_variable('w2_a', [self.a_dim, HIDDEN_2_SIZE], initializer=init_w)
                b2 = tf.get_variable('b2', [1, HIDDEN_2_SIZE], initializer=init_b)
                hidden_2 = tf.nn.elu(tf.matmul(hidden_1, w2_s) + tf.matmul(a, w2_a) + b2)
            q = tf.layers.dense(hidden_2, 1, kernel_initializer=init_w, bias_initializer=init_b, name='q')

        return q


    def learn(self, s, a, r, s_):
        self.sess.run([self.train_op, self.target_update], feed_dict={self.s: s, self.a: a,
                                                                      self.r: r, self.s_: s_,
                                                                      self.is_training: True})
