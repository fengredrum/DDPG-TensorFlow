import gym
from actor_network import *
from critic_network import *
from replayBuffer import Memory

# Hyper Parameters:
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32

OUTPUT_GRAPH = False


class DDPG:
    """docstring for DDPG"""

    def __init__(self, env):
        self.environment = env
        self.environment.seed(1)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high

        with tf.name_scope('inputs'):
            state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='s')
            reward = tf.placeholder(tf.float32, [None, 1], name='r')
            next_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='s_')
            is_training = tf.placeholder(tf.bool)

        self.sess = tf.InteractiveSession()

        # Create actor and critic.
        self.actor = Actor(self.sess, BATCH_SIZE, self.action_dim, self.action_bound, state, next_state, is_training)
        self.critic = Critic(self.sess, BATCH_SIZE, self.state_dim, self.action_dim,
                             self.actor.a, self.actor.a_, state, reward, next_state, is_training)
        self.actor.add_grad_to_graph(self.critic.a_grads)

        self.sess.run(tf.global_variables_initializer())

        self.M = Memory(REPLAY_BUFFER_SIZE, dims=2 * self.state_dim + self.action_dim + 1)

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def train(self):
        b_M = self.M.sample(BATCH_SIZE)
        b_s = b_M[:, :self.state_dim]
        b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
        b_r = b_M[:, -self.state_dim - 1: -self.state_dim]
        b_s_ = b_M[:, -self.state_dim:]

        self.critic.learn(b_s, b_a, b_r, b_s_)
        self.actor.learn(b_s, b_s_)

    # def noise_action(self,state):
    #     # Select action a_t according to the current policy and exploration noise
    #     action = self.actor_network.action(state)
    #     return action+self.exploration_noise.noise()

    def action(self, state):
        action = self.actor.choose_action(state)
        return action

    def perceive(self, state, action, reward, next_state):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.M.store_transition(state, action, reward, next_state)

        # Store transitions to replay start size then start training
        if self.M.pointer > REPLAY_BUFFER_SIZE:
            self.train()
