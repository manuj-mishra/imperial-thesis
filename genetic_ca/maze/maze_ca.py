import random
from collections import defaultdict, deque
import numpy as np
from scipy.signal import convolve2d
from media import save_image, init_image
from util.grid import near
from util.maze import intmap, bfs


class MazeCA:
  def __init__(self, B, S):
    self.nx, self.ny = 32, 32
    self.X = np.zeros((self.ny, self.nx))
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
    return np.isin(n, self.B) | (self.X & np.isin(n, self.S))

  def generate(self, n_iter=30, media=False):
    folder, ax = None, None
    if media:
      folder = "temp/gen_frames"
      ax = init_image()

    # Maze size
    nx, ny = 32, 32
    self.X = np.random.random((ny, nx)) > 0.75

    frames_per_image = 1

    for i in range(n_iter):
      self.X = self._step()
      if not i % frames_per_image and media:
        # print('{}/{}'.format(i, n_iter))
        Y = intmap(self.X)
        save_image(Y, i, ax, folder=folder)

    self.X = intmap(self.X)
    if media:
      save_image(self.X, n_iter, ax, folder=folder)

  def find_regions(self, media=False):
    # cells = {(x,y):region}
    # regions = {region:set((x1, y1), ... , (xn, yn))}
    folder, ax = None, None
    if media:
      folder = "temp/reg_frames"
      ax = init_image()

    cells = dict()
    regions = defaultdict(set)

    spaces = set()
    for i in range(self.nx):
      for j in range(self.ny):
        if self.X[i][j] == 0:
          spaces.add((i, j))

    start = (self.nx - 1, 0)
    r1 = bfs(start, self.X, self.nx)
    M_copy = self.X.copy()
    regnum = 1
    for cell in r1:
      M_copy[cell[0], cell[1]] = 3
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
      for cell in reg:
        M_copy[cell[0], cell[1]] = 2
      if media:
        save_image(M_copy, regnum, ax, folder=folder)
      reg = bfs(start, self.X, self.nx)
      for cell in reg:
        M_copy[cell[0], cell[1]] = 3
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
    for i in range(3000):
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
    print("MAXED OUT MERGE")
    return False

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
