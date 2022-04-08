import numpy as np
from scipy.signal import convolve2d
from media import save_image

class SlimeCA:
  def __init__(self, chromosome):
    self.nx, self.ny = 20, 20
    self.X = np.zeros((self.ny, self.nx))
    self.C = chromosome
    self.n_food = 3
    self.food_locs = []

  def _step(self):
    sum_filter = np.ones((3, 3))
    # sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # sobel_y = sobel_x.T

    slime_mask = self.X == 1
    food_mask = self.X == 2

    n_food = convolve2d(food_mask, sum_filter, mode='same', boundary='wrap') - food_mask
    n_slime = convolve2d(slime_mask, sum_filter, mode='same', boundary='wrap') - slime_mask

    survived = slime_mask * (
        (self.C.s_coefs['n_food_c'] * n_food) + (self.C.s_coefs['n_slime_c'] * n_slime)
    )
    born = ~slime_mask * (n_slime > 0) * (
        (self.C.b_coefs['n_food_c'] * n_food) + (self.C.b_coefs['n_slime_c'] * n_slime)
    )
    new_slime = ((survived + born) / 16) > 0.1
    self.X = np.maximum(food_mask * 2, new_slime)

  def run(self, n_iter=50, media=False, folder="temp/gen_frames", ax=None):

    self.X[self.nx // 2 - 1: self.nx // 2 + 1, self.ny // 2 - 1: self.ny // 2 + 1] = 1
    centre = np.array([[self.nx // 2 - 1, self.nx // 2 - 1, self.nx // 2, self.nx // 2],
                      [self.ny // 2 - 1, self.ny // 2, self.ny // 2 - 1, self.ny // 2]])
    centre_ixs = np.ravel_multi_index(centre, self.X.shape)
    possible_food_ixs = [i for i in np.arange(self.X.size) if i not in centre_ixs]
    food_ixs = np.random.choice(possible_food_ixs, replace=False, size=self.n_food)
    self.food_locs = np.unravel_index(food_ixs, self.X.shape)
    self.X[self.food_locs] = 2

    frames_per_image = 1

    for i in range(n_iter):
      self._step()
      if not i % frames_per_image and media:
        save_image(self.X, i, ax, folder=folder)

    if media:
      save_image(self.X, n_iter, ax, folder=folder)

    return self.X

  def food_reached(self):
    count = 0
    fxs, fys = self.food_locs
    for i in range(self.n_food):
      for x, y in near(fxs[i], fys[i], self.X.shape[0]):
        count += self.X[x, y] == 1
    return count

  def slime_size(self):
    return np.sum(self.X == 1)


def near(x, y, n):
  """Finds valid neighbours of (x,y) in grid of size nxn"""
  adj = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
  return [(i, j) for (i, j) in adj if 0 <= i < n and 0 <= j < n]
