"""
A simple demo of pendulum control using ddpg algorithm.
Feel free to use the code.
If you like it, please follow me or give me a star.
If you don't, any comment is welcome.
Just open an issue in my repository, I'll handle it ASAP.
There will be more deep RL algorithm coming soon.
"""

from ddpg import *

import matplotlib.pyplot as plt
import seaborn as sns
import time

start = time.clock()
np.random.seed(1)
tf.set_random_seed(1)

# hyper parameters
MAX_EPISODES = 2000
MAX_EP_STEPS = 200

RENDER = False

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
agent = DDPG(env)

var = 3  # control exploration
episode_reward = []
for i in range(MAX_EPISODES+1):
    s = env.reset()
    ep_reward = 0
    if var >= 0.1:
        var *= .985  # decay the action randomness

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        # Add exploration noise
        a = agent.action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)
        agent.perceive(s, a, r, s_)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            episode_reward.append(ep_reward)
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > 10:
                RENDER = True
            break

agent.sess.close()
end = time.clock()
print('Running time: %s Seconds' % (end - start))

plt.figure(1)
sns.set(style="darkgrid")
plt.plot(episode_reward, label='DDPG')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='best')

plt.show()
