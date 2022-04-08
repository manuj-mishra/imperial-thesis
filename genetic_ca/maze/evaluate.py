from collections import deque
from maze.maze_ca import near

#TODO: Eventually move this all to maze.py

def dead_ends_and_path_length(M):
  size = M.shape[0]
  q = deque([(size - 1, 0, 0)])
  visited = set()
  dead_ends = 0
  result = None
  while q:
    curr_x, curr_y, curr_len = q.popleft()
    if (curr_x, curr_y) not in visited:
      visited.add((curr_x, curr_y))
      adj = near(curr_x, curr_y, size)
      if all((x, y) in visited for x, y in adj):
        dead_ends += 1
        continue

      for x, y in adj:
        if M[x, y] == 0 and (x, y) not in visited:
          q.append((x, y, curr_len + 1))
        if M[x, y] == 4:
          result = M, dead_ends, curr_len + 1
  return result


def find_sol_path(M):
  size = M.shape[0]
  q = deque([(size - 1, 0, 2)])
  visited = set()
  while q:
    curr_x, curr_y, curr_len = q.popleft()
    if (curr_x, curr_y) not in visited:
      visited.add((curr_x, curr_y))
      M[curr_x, curr_y] = curr_len
      adj = near(curr_x, curr_y, size)
      for x, y in adj:
        if M[x, y] == 0 and (x, y) not in visited:
          q.append((x, y, curr_len + 1))
        if M[x, y] == 4:
          M[x, y] = curr_len + 1
          print(M)
          return find_path_from_end(M)


def find_path_from_end(M):
  n = M.shape[0]
  cx, cy = 0, n - 1
  while M[cx, cy] > 2:
    curr = M[cx, cy]
    M[cx, cy] = 2
    cx, cy = [(x, y) for x, y in near(cx, cy, n) if M[x, y] == curr - 1][0]
  for i in range(n):
    for j in range(n):
      if M[i, j] > 2:
        M[i, j] = 0
  return M

# def dead_ends(M):
# # Number of dead ends using DFS
#   size = M.shape[0]
#   s = deque([(size - 1, 0)])
#   visited = set()
#   count = 0
#   while s:
#     curr_x, curr_y = s.pop()
#     if (curr_x, curr_y) not in visited:
#       visited.add((curr_x, curr_y))
#       adj = near(curr_x, curr_y, size)
#       is_leaf = True
#       for x, y in adj:
#         if M[x, y] == 0 and (x, y) not in visited:
#           is_leaf = False
#           s.append((x, y))
#       if is_leaf:
#         M[curr_x, curr_y] = 2
#         count += 1
#   return M, count
