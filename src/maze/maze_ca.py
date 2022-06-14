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


class MazeCA:
  def __init__(self, B, S):
    self.nx, self.ny = GRID_SIZE, GRID_SIZE
    self.X = np.random.random((self.ny, self.nx)) > INITIAL_DENSITY
    self.B = B
    self.S = S

  def run(self, media=False):
    self.generate(media=media)
    cells, regions = self.find_regions(media=media)
    success = self.merge_regions(cells, regions, media=media)
    return success

  def _step(self):
    K = np.ones((3, 3))
    n = convolve2d(self.X, K, mode='same', boundary='wrap') - self.X
    return (~self.X & np.isin(n, self.B)) | (self.X & np.isin(n, self.S))

  def generate(self, media=False):
    folder, ax = None, None
    if media:
      folder = "temp/gen_frames"
      fig, ax = init_image()
    frames_per_image = 1

    for i in range(GEN_ITERS):
      self.X = self._step()
      if not i % frames_per_image and media:
        # print('{}/{}'.format(i, n_iter))
        Y = intmap(self.X)
        save_image(Y, i, ax, folder=folder)

    self.X = intmap(self.X)
    if media:
      save_image(self.X, GEN_ITERS, ax, folder=folder)

  def find_regions(self, media=False):
    # cells = {(x,y):region}
    # regions = {region:set((x1, y1), ... , (xn, yn))}
    folder, ax, M_copy = None, None, None
    if media:
      folder = "temp/reg_frames"
      ax = init_image()
      M_copy = self.X.copy()

    cells = dict()
    regions = defaultdict(set)

    spaces = set()
    for i in range(self.nx):
      for j in range(self.ny):
        if self.X[i][j] == 0:
          spaces.add((i, j))

    start = (self.nx - 1, 0)
    r1 = bfs(start, self.X, self.nx)
    regnum = 1
    if media:
      for cell in r1:
        M_copy[cell[0], cell[1]] = 3
    for cell in r1:
      cells[cell] = regnum
      regions[regnum].add(cell)
    spaces.difference_update(r1)

    if media:
      save_image(M_copy, regnum, ax, folder=folder)

    reg = r1
    n_iter = 0
    while spaces:
      regnum += 1
      start = spaces.pop()
      if media:
        for cell in reg:
          M_copy[cell[0], cell[1]] = 2
        save_image(M_copy, regnum, ax, folder=folder)
      reg = bfs(start, self.X, self.nx)
      if media:
        for cell in reg:
          M_copy[cell[0], cell[1]] = 3
      for cell in reg:
        cells[cell] = regnum
        regions[regnum].add(cell)
      spaces.difference_update(reg)
      n_iter += 1

    return cells, regions

  def merge_regions(self, cells, regions, media=False):
    folder, ax = None, None
    if media:
      folder = "temp/mer_frames"
      ax = init_image()
    curr = regions[1]
    for i in range(GRID_SIZE ** 2):
      fringe = set().union(*(near(c[0], c[1], self.nx) for c in curr)) - curr
      if (0, self.ny - 1) in fringe:
        if media:
          save_image(self.X, i, ax, folder=folder)
        return True

      if media:
        M_copy = self.X.copy()
        for x, y in curr:
          M_copy[x, y] = 2
        for x, y in fringe:
          M_copy[x, y] = 3
        save_image(M_copy, i, ax, folder=folder)

      cands = []
      for f in fringe:
        zeros = set(z for z in near(f[0], f[1], self.nx) if self.X[z[0], z[1]] == 0)
        if len(zeros - curr) > 0:
          cands.append(f)
      if len(cands) > 0:
        cx, cy = random.choice(cands)
      else:
        return False
      curr.add((cx, cy))
      self.X[cx, cy] = 0
      new_regs = [cells[around] for around in near(cx, cy, self.nx) if around in cells]
      curr = curr.union(*(regions[r] for r in new_regs))
    raise Exception("Maximum merge limit reached")

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
