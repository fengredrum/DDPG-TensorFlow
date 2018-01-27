# DDPG-TensorFlow

This is an tensorflow implementation of the paper "Continuous control with deep reinforcement learning".
The code is adapted from [here](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow) with some improvement. The main difference between the MorvanZhou's version and this code is
1. Using `tf.train.ExponentialMovingAverage` to soft update the target network instead of `tf.assign`;
2. Changing the activation function `tf.nn.relu` to `tf.nn.elu`;
3. Using `tf.train.AdagradOptimizer` to train the neural network;
4. Batch Updating the parameters using accumulating averaged gradients instead of the mean of mini batch.
The speed is 4.0X faster than the MorvanZhou's version when running on the same desktop computer with GTX 960 GPU, and achieving  more stable learning curve.
Typing `python Train.py` in the terminal to run, make sure you've already installed tensorflow and open ai gym.
Any comment or suggestion is welcome, you can open an issue or contact me via "fengredrum@gmail.com".
More deep RL algorithm is coming soon!

## Reference
1. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971.
2. https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
3. https://github.com/songrotek/DDPG
4. https://github.com/RuiShu/micro-projects/tree/master/tf-ema


