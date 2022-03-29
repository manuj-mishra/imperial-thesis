from collections import deque

from generator.region_merge import near

def solution_length(M):
  size = M.shape[0]
  q = deque([(size-1, 0, 0)])
  visited = set()
  while q:
    curr_x, curr_y, curr_n = q.popleft()
    visited.add((curr_x, curr_y))
    adj = near(curr_x, curr_y, size)
    for x, y in adj:
      if M[x, y] == 0 and (x, y) not in visited:
        q.append((x, y, curr_n + 1))
      if M[x, y] == 4:
        print("Max path length:", curr_n + 1)
        return curr_n + 1
