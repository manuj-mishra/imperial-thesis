import numpy as np
from scipy.signal import convolve2d
from media import make_files, intmap, init_image, save_image


def ca_step(X, B, S):
  K = np.ones((3, 3))
  n = convolve2d(X, K, mode='same', boundary='wrap') - X
  return np.isin(n, B) | (X & np.isin(n, S))
  # return (n == 3) | (X & ((n > 0) & (n < 5)))


def generate_maze(B, S, n_iter=150, folder="gen_frames"):
  # Maze size
  nx, ny = 44, 44
  X = np.zeros((ny, nx), dtype=np.bool)
  # Size of initial random area (must be even numbers)
  mx, my = 6, 6

  # Initialize a patch with a random mx x my region
  r = np.random.random((my, mx)) > 0.75
  X[ny // 2 - my // 2:ny // 2 + my // 2, nx // 2 - mx // 2:nx // 2 + mx // 2] = r

  frames_per_image = 2

  ax = init_image()
  for i in range(n_iter):
    X = ca_step(X, B, S)
    if not i % frames_per_image:
      # print('{}/{}'.format(i, n_iter))
      Y = intmap(X)
      save_image(Y, i, ax, folder=folder)

  save_image(intmap(X), n_iter, ax, folder=folder)
  return intmap(X)
