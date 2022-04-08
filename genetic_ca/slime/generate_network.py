import numpy as np
from scipy.signal import convolve2d
from media import make_files, intmap, init_image, save_image


def ca_step(X, B_chrom, S_chrom):
  sum_filter = np.ones((3, 3))
  # sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  # sobel_y = sobel_x.T

  slime_mask = X[X == 1].astype(int)
  food_mask = X[X == 2].astype(int)

  n_food = convolve2d(food_mask, sum_filter, mode='same', boundary='wrap') - food_mask
  n_slime = convolve2d(slime_mask, sum_filter, mode='same', boundary='wrap') - slime_mask

  survived = slime_mask * ((S_chrom['n_food_c'] * n_food) + (S_chrom['n_slime_c'] * n_slime))
  born = ~slime_mask * n_slime[n_slime > 0] * ((B_chrom['n_food_c'] * n_food) + (B_chrom['n_slime_c'] * n_slime))
  new_X = (survived + born) / 16
  return np.maximum(food_mask * 2, new_X)


def generate_maze(B, S, n_iter=150, media=False, folder="temp/gen_frames", ax=None):
  # Maze size
  nx, ny = 32, 32
  X = np.zeros((ny, nx), dtype=np.bool)
  food_ixs = np.random.choice(np.arange(X.size), replace=False, size=3)
  X[food_ixs] = 2

  frames_per_image = 2

  for i in range(n_iter):
    X = ca_step(X, B, S)
    if not i % frames_per_image and media:
      # print('{}/{}'.format(i, n_iter))
      Y = intmap(X)
      save_image(Y, i, ax, folder=folder)

  if media:
    save_image(intmap(X), n_iter, ax, folder=folder)
  return intmap(X)
