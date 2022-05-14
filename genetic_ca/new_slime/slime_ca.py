import random
from collections import defaultdict, deque

import numpy as np
from scipy.signal import convolve2d

from util.media import save_image, init_image, make_files, clear_temp_folders
from util.grid import near
from util.maze import intmap, bfs

GRID_SIZE = 32
GEN_ITERS = 50
INITIAL_DENSITY = 0.75


class SlimeCA:
  def __init__(self, b, s, n_food=3):
    self.nx, self.ny = GRID_SIZE, GRID_SIZE
    self.X = np.zeros((self.ny, self.nx))
    self.b = b
    self.s = s

    self.n_food = n_food
    self.food_locs = []

  def _step(self):
    K = np.ones((3, 3))

    slime_mask = self.X == 1
    food_mask = self.X == 2

    n_food = convolve2d(food_mask, K, mode='same', boundary='wrap') - food_mask
    n_slime = convolve2d(slime_mask, K, mode='same', boundary='wrap') - slime_mask
    n_food = n_food.astype(int)
    n_slime = n_slime.astype(int)
    birth = (np.take(self.b, n_food) >> n_slime) & 1
    survival = (np.take(self.s, n_food) >> n_slime) & 1
    return (~slime_mask & birth) | (slime_mask & survival)

  def run(self, media=False):
    folder, ax = None, None
    if media:
      folder = "temp/gen_frames"
      ax = init_image()

    self.X[self.nx // 2 - 1: self.nx // 2 + 1, self.ny // 2 - 1: self.ny // 2 + 1] = 1
    centre = np.array([[self.nx // 2 - 1, self.nx // 2 - 1, self.nx // 2, self.nx // 2],
                       [self.ny // 2 - 1, self.ny // 2, self.ny // 2 - 1, self.ny // 2]])
    centre_ixs = np.ravel_multi_index(centre, self.X.shape)

    possible_food_ixs = [i for i in np.arange(self.X.size) if i not in centre_ixs]
    food_ixs = np.random.choice(possible_food_ixs, replace=False, size=self.n_food)
    self.food_locs = np.unravel_index(food_ixs, self.X.shape)
    self.X[self.food_locs] = 2

    frames_per_image = 1

    for i in range(GEN_ITERS):
      self._step()
      if not i % frames_per_image and media:
        save_image(self.X, i, ax, folder=folder)

    if media:
      save_image(self.X, GEN_ITERS, ax, folder=folder)

    return self.X

  def metrics(self):
    size_ratio = np.sum(self.X == 1) / self.X.size
    return size_ratio, self.food_ratio()

  def food_ratio(self):
    count = 0
    total = 0
    fxs, fys = self.food_locs
    for i in range(self.n_food):
      for x, y in near(fxs[i], fys[i], self.nx):
        count += self.X[x, y] == 1
        total += 1
    return count / total

  def save_experiment(self, rname):
    X = self.run(media=True)
    make_files(final_state=X, fname="generation", rname=rname, clear=True)
    size, food = self.metrics()
    print("SIZE", size)
    print("FOOD", food)
    clear_temp_folders()



