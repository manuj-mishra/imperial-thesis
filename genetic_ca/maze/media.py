import glob
import os
import shutil

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from maze.evaluate import dead_ends_and_path_length
from maze.maze import generate_maze, get_regions, region_merge

root = "."

"""
0 = space
1 = wall
2 = start / visited
3 = fringe
4 = goal
"""


def save(B, S, exp_n):
  B.sort()
  S.sort()

  # Rulestring is in B/S notation (Birth/Survival)
  rulestring = ''.join(str(i) for i in B) + '_' + ''.join(str(i) for i in S)
  print(f"Running CA {rulestring}")
  rulestring = f"{exp_n}/{rulestring}"

  ax = init_image()
  X = generate_maze(B, S, media=True, folder='temp/gen_frames', ax=ax)
  make_files(frame_folder='gen_frames', rstring=rulestring, name="generation", final_state=X, clear=True)

  ax = init_image()
  cells, regions, M = get_regions(X, media=True, folder='temp/reg_frames', ax=ax)
  make_files(frame_folder='reg_frames', rstring=rulestring, name="regions", final_state=M, clear=False)

  ax = init_image()
  M, success = region_merge(regions, cells, M, media=True, folder='temp/merge_frames', ax=ax)
  make_files(frame_folder='merge_frames', rstring=rulestring, name="merging", final_state=M, clear=False)

  if success:
    _, ends, length = dead_ends_and_path_length(M)
    print("Solution length:", length)
    print("Dead ends:", ends)
    # M = find_sol_path(M)
    # save_final_image(M, path=f'./out/{rulestring}/solution.png', ax=init_image())
  else:
    print("Region merge failed")
  clear_temp_folders()


def make_files(rstring, name, frame_folder=None, final_state=None, clear=True):
  dirname = f"{root}/out/{rstring}"

  dirs = ["gifs", "final_frames", "np_arrays"]
  if os.path.exists(dirname):
    if clear:
      shutil.rmtree(dirname)
      for dir in dirs:
        os.makedirs(f"{dirname}/{dir}")
  else:
    for dir in dirs:
      os.makedirs(f"{dirname}/{dir}")

  if frame_folder is not None:
    frames = [Image.open(image) for image in sorted(glob.glob(f"{root}/temp/{frame_folder}/*.png"))]
    frames[0].save(f"{dirname}/gifs/{name}.gif", format="GIF", append_images=frames[1:],
                   save_all=True, duration=50)
    frame_last = frames[-1]
    frame_last.save(f"{dirname}/final_frames/{name}.png")

  if final_state is not None:
    fname = f"{dirname}/np_arrays/{name}.npy"
    with open(fname, 'wb') as f:
      np.save(f, final_state)


def init_image(width=500, height=500, dpi=10):
  fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
  ax = fig.add_subplot(111)
  return ax


def save_image(M, i, ax, folder, cmap=ListedColormap(["w", "k", "y", "g", "r"]), dpi=10):
  im = ax.imshow(M, cmap=cmap, interpolation='nearest')
  plt.axis('off')
  plt.savefig('./{:s}/_img{:04d}.png'.format(folder, i), dpi=dpi)
  plt.cla()


def save_final_image(M, path, ax, cmap=ListedColormap(["w", "k", "y", "g", "r"]), dpi=10):
  im = ax.imshow(M, cmap=cmap, interpolation='nearest')
  plt.axis('off')
  plt.savefig(f'./{path}', dpi=dpi)
  plt.cla()


def clear_temp_folders():
  parent = f"{root}/temp"
  dirs = next(os.walk(parent))[1]
  for dir in dirs:
    droot, _, files = next(os.walk(f"{parent}/{dir}"))
    for file in files:
      os.remove(os.path.join(droot, file))
