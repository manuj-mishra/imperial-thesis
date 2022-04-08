import random
from collections import defaultdict, deque
import numpy as np
from scipy.signal import convolve2d
from media import save_image


def ca_step(X, B, S):
  K = np.ones((3, 3))
  n = convolve2d(X, K, mode='same', boundary='wrap') - X
  return np.isin(n, B) | (X & np.isin(n, S))


def generate_maze(B, S, n_iter=150, media=False, folder="temp/gen_frames", ax=None):
  # Maze size
  nx, ny = 32, 32
  X = np.random.random((ny, nx)) > 0.75

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


def get_regions(M, media=False, folder="temp/reg_frames", ax=None):
  # cells = {(x,y):region}
  # regions = {region:set((x1, y1), ... , (xn, yn))}

  n = M.shape[0]

  cells = dict()
  regions = defaultdict(set)

  spaces = set()
  for i in range(n):
    for j in range(n):
      if M[i][j] == 0:
        spaces.add((i, j))

  start = (n - 1, 0)
  r1 = bfs(start, M, n)
  M_copy = M.copy()
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
    reg = bfs(start, M, n)
    for cell in reg:
      M_copy[cell[0], cell[1]] = 3
      cells[cell] = regnum
      regions[regnum].add(cell)
    spaces.difference_update(reg)
    n_iter += 1
  return cells, regions, M


def region_merge(regions, cells, M, media=False, folder="temp/merge_frames", ax=None):
  curr = regions[1]
  n = M.shape[0]
  for i in range(3000):
    fringe = set().union(*(near(c[0], c[1], n) for c in curr)) - curr

    if (0, n - 1) in fringe:
      if media:
        save_image(M, i, ax, folder=folder)
      return M, True

    if media:
      M_copy = M.copy()
      for x, y in curr:
        M_copy[x, y] = 2
      for x, y in fringe:
        M_copy[x, y] = 3
      save_image(M_copy, i, ax, folder=folder)

    cands = []
    for f in fringe:
      zeros = set(z for z in near(f[0], f[1], n) if M[z[0], z[1]] == 0)
      if len(zeros - curr) > 0:
        cands.append(f)
    if len(cands) > 0:
      cx, cy = random.choice(cands)
    else:
      return M, False
    curr.add((cx, cy))
    M[cx, cy] = 0
    new_regs = [cells[around] for around in near(cx, cy, n) if around in cells]
    curr = curr.union(*(regions[r] for r in new_regs))
  print("MAXED OUT MERGE")
  return M, False


# AUXILIARY FUNCTIONS

def intmap(X):
  """Converts bool maze to int. Sets start and goal."""
  Y = X.astype(int)
  Y[Y.shape[0] - 1, 0] = 2
  Y[0, Y.shape[1] - 1] = 4
  return Y


def near(x, y, n):
  """Finds valid neighbours of (x,y) in maze of size nxn"""
  adj = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
  return [(i, j) for (i, j) in adj if 0 <= i < n and 0 <= j < n]


def bfs(start, M, n):
  """Breadth first search in maze M of size nxn from cell start"""
  q = deque([start])
  visited = set()
  while q:
    curr = q.popleft()
    if curr not in visited:
      visited.add(curr)
      adj = near(curr[0], curr[1], n)
      for x, y in adj:
        if M[x, y] == 0 and (x, y) not in visited:
          q.append((x, y))
  return visited
