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
  def __init__(self, C):
    self.nx, self.ny = GRID_SIZE, GRID_SIZE
    self.X = np.random.random((self.ny, self.nx)) > INITIAL_DENSITY
    self.C = C

  def _step(self):
    K = np.ones((3, 3))

    slime_mask = self.X == 1
    food_mask = self.X == 2

    n_food = convolve2d(food_mask, K, mode='same', boundary='wrap') - food_mask
    n_slime = convolve2d(slime_mask, K, mode='same', boundary='wrap') - slime_mask

    b =
    return (~self.X & np.isin(n, self.B)) | (self.X & np.isin(n, self.S))

  def run(self, media=False):
    folder, ax = None, None
    if media:
      folder = "temp/gen_frames"
      ax = init_image()
    frames_per_image = 2

    for i in range(GEN_ITERS):
      self.X = self._step()
      if not i % frames_per_image and media:
        # print('{}/{}'.format(i, n_iter))
        save_image(self.X, i, ax, folder=folder)

    if media:
      save_image(self.X, GEN_ITERS, ax, folder=folder)

  def metrics(self, media=False):
    folder, ax, M_copy = None, None, None
    if media:
      folder = "temp/eva_frames"
      ax = init_image()
      M_copy = self.X.copy()

    q = deque([(self.nx - 1, 0, 0)])
    visited = set()
    dead_ends = 0
    path_len = None

    while q:
      curr_x, curr_y, curr_len = q.popleft()
      if (curr_x, curr_y) not in visited:
        visited.add((curr_x, curr_y))

        if media:
          M_copy[curr_x, curr_y] = 2

        adj = near(curr_x, curr_y, self.nx)
        if all(self.X[x, y] == 1 or (x, y) in visited for x, y in adj):
          dead_ends += 1

          if media:
            M_copy[curr_x, curr_y] = 3
            save_image(M_copy, dead_ends, ax, folder=folder)

          continue

        for x, y in adj:
          if (x, y) not in visited:
            if self.X[x, y] == 0:
              q.append((x, y, curr_len + 1))
            elif self.X[x, y] == 4:
              visited.add((x, y))
              path_len = curr_len + 1

    reachable = len(visited) / self.X.size
    return dead_ends, path_len, reachable

  def save_experiment(self, rname, attempt=1):
    if attempt == 0:
      print(f"Running CA {rname}")

    self.generate(media=True)
    make_files(final_state=self.X, fname="generation", rname=rname, clear=True)

    cells, regions, = self.find_regions(media=True)
    make_files(final_state=self.X, fname="regions", rname=rname)

    success = self.merge_regions(cells, regions, media=True)
    make_files(final_state=self.X, fname="merging", rname=rname)

    if success:
      ends, length, reachable = self.metrics(media=True)
      make_files(final_state=self.X, fname="evaluation", rname=rname)
      print("Dead ends:", ends)
      print("Solution length:", length)
      print("Reachable:", reachable)
      # M = find_sol_path(M)
      # save_final_image(M, path=f'./out/{rulestring}/solution.png', ax=init_image())
    else:
      print(f"Attempt {attempt}: Region merge failed")
      if attempt < 3:
        print("Trying again")
        self.save_experiment(rname, attempt=attempt + 1)
    clear_temp_folders()
