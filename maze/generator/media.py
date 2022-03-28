import glob
import os
import shutil

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

root = "."

"""
0 = space
1 = wall
2 = start / visited
3 = fringe
4 = goal
"""

def intmap(X):
  """
  Converts boolean maze matrix into ints and marks start and goal positions.
  :param list X: A 2D boolean matrix representing the maze
  :return: A 2D int matrix with start and goal marked
  :rtype: list
  """
  Y = X.astype(int)
  Y[Y.shape[0] - 1, 0] = 2
  Y[0, Y.shape[1] - 1] = 4
  return Y


def init_image(width=500, height=500, dpi=10):
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
  ax = fig.add_subplot(111)
  return ax

def save_image(M, i, ax, folder, cmap=ListedColormap(["w", "k", "y", "g", "r"]), dpi=10):
  im = ax.imshow(M, cmap=cmap, interpolation='nearest')
  plt.axis('off')
  plt.savefig('./{:s}/_img{:04d}.png'.format(folder, i), dpi=dpi)
  plt.cla()

def clear_temp_folders():
  parent = f"{root}"
  dirs = ["gen_frames", "reg_frames", "merge_frames"]
  for dir in dirs:
    for file in os.scandir(f"{parent}/{dir}"):
      os.remove(file.path)

def make_files(frame_folder, rstring, name, final_state=None, clear=True):
  dirname = f"{root}/{rstring}"

  dirs = ["gifs", "final_frames", "np_arrays"]
  if os.path.exists(dirname):
    if clear:
      shutil.rmtree(dirname)
      for dir in dirs:
        os.makedirs(f"{dirname}/{dir}")
  else:
    for dir in dirs:
      os.makedirs(f"{dirname}/{dir}")

  frames = [Image.open(image) for image in sorted(glob.glob(f"{root}/{frame_folder}/*.png"))]
  frame_one = frames[0]
  frame_one.save(f"{dirname}/gifs/{name}.gif", format="GIF", append_images=frames,
                 save_all=True, duration=50)
  frame_last = frames[-1]
  frame_last.save(f"{dirname}/final_frames/{name}.png")

  if final_state is not None:
    fname = f"{dirname}/np_arrays/{name}.npy"
    with open(fname, 'wb') as f:
      np.save(f, final_state)