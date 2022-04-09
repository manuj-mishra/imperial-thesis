def near(x, y, n):
  """Finds valid neighbours of (x,y) in grid of size nxn"""
  adj = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
  return [(i, j) for (i, j) in adj if 0 <= i < n and 0 <= j < n]