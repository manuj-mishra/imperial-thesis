import math

import tensorflow as tf
from matplotlib.colors import ListedColormap
from tensorflow import keras

import numpy as np

from neural_slime.slimenet_keras import SlimeNetKeras
from util.media import save_final_image, init_image

"""
food / other road    = -1 (black)
road                 = 0  (white)
shortest path        = 1  (green)
"""
class NeuralSlimeCA:
  def __init__(self):
    self.nx, self.ny = 20, 20
    self.X = np.zeros((16, self.ny, self.nx))
    self.Y = np.full((self.ny, self.nx), -1)
    self.food_locs = []
    self.loss = 0

  def initialise_state(self, n_food):
    # Place food randomly in first layer
    possible_food_ixs = [i for i in np.arange(self.nx * self.ny)]
    food_ixs = np.random.choice(possible_food_ixs, replace=False, size=n_food)
    visual = np.zeros_like(self.X[0])
    self.food_locs = np.unravel_index(food_ixs, visual.shape)
    visual[self.food_locs] = -1
    save_final_image(visual, 'food', init_image(), cmap=ListedColormap(['k', 'w']))
    self.X[0] = visual

  def initialise_goal(self):
    # Generate minimum single trunk steiner tree
    ys, xs = self.food_locs
    med = math.floor(np.median(ys))
    self.Y[med, np.min(xs):np.max(xs) + 1] = 1
    for i in range(len(xs)):
      if ys[i] < med:
        self.Y[ys[i]:med + 1, xs[i]] = 1
      else:
        self.Y[med:ys[i] + 1, xs[i]] = 1
    save_final_image(self.Y, 'tree', init_image(), cmap=ListedColormap(['k', 'w', 'g']))

if __name__ == "__main__":
  inputs = keras.Input(shape=(16, 5))
  x = keras.layers.Flatten()(inputs)
  x = keras.layers.Dense(64, activation='relu')(x)
  outputs = keras.layers.Dense(16)(x)
  model = SlimeNetKeras(inputs=inputs, outputs=outputs)

  nsca = NeuralSlimeCA()
  nsca.initialise_state(6)
  nsca.initialise_goal()
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
  model.fit(np.expand_dims(nsca.X, axis=0), np.expand_dims(nsca.Y, axis=0), epochs=4)
