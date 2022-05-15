from collections import deque
from util.grid import near

def intmap(X):
  """Converts bool maze to int. Sets start and goal."""
  Y = X.astype(int)
  Y[Y.shape[0] - 1, 0] = 2
  Y[0, Y.shape[1] - 1] = 4
  return Y


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