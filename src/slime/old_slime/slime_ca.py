import numpy as np
from scipy.signal import convolve2d
from util.media import save_image, init_image, make_files, clear_temp_folders
from util.grid import near


class SlimeCA:
  def __init__(self, chromosome, n_food = 3):
    self.nx, self.ny = 20, 20
    self.X = np.zeros((self.ny, self.nx))
    self.C = chromosome
    self.n_food = n_food
    self.food_locs = []

  def _step(self):
    sum_filter = np.ones((3, 3))
    x_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    y_sobel = x_sobel.T

    slime_mask = self.X == 1
    food_mask = self.X == 2

    n_food = convolve2d(food_mask, sum_filter, mode='same', boundary='wrap') - food_mask
    x_food = convolve2d(food_mask, x_sobel, mode='same', boundary='wrap')
    y_food = convolve2d(food_mask, y_sobel, mode='same', boundary='wrap')
    n_slime = convolve2d(slime_mask, sum_filter, mode='same', boundary='wrap') - slime_mask
    x_slime = convolve2d(slime_mask, x_sobel, mode='same', boundary='wrap')
    y_slime = convolve2d(slime_mask, y_sobel, mode='same', boundary='wrap')

    survived = slime_mask * (
        (self.C.coefs['s_food_n'] * n_food) + (self.C.coefs['s_slime_n'] * n_slime) +
        (self.C.coefs['s_food_x'] * x_food) + (self.C.coefs['s_slime_x'] * x_slime) +
        (self.C.coefs['s_food_y'] * y_food) + (self.C.coefs['s_slime_y'] * y_slime)
    )
    born = ~slime_mask * (n_slime > 0) * (
        (self.C.coefs['b_food_n'] * n_food) + (self.C.coefs['b_slime_n'] * n_slime) +
        (self.C.coefs['b_food_x'] * x_food) + (self.C.coefs['b_slime_x'] * x_slime) +
        (self.C.coefs['b_food_y'] * y_food) + (self.C.coefs['b_slime_y'] * y_slime)
    )
    new_slime = ((survived + born + 16) / 48) > self.C.coefs['threshold']
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
      for x, y in near(fxs[i], fys[i], self.nx):
        count += self.X[x, y] == 1
    return count

  def slime_size(self):
    return np.sum(self.X == 1)

  def save_experiment(self, rname):
    ax = init_image()
    X = self.run(media=True, folder='temp/gen_frames', ax=ax)
    make_files(final_state=X, fname="generation", rname=rname, clear=True)
    print("SLIME SIZE:", self.slime_size())
    print("FOOD REACHED:", self.food_reached())
    clear_temp_folders()
