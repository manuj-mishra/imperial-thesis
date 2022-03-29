import random


class Rulestring:
  def __init__(self, b, s):
    self.b = b
    self.s = s
    self.rstring = self._rstring(b, s)

  def _rstring(self, b, s):
    rstring = 0

    for i in range(1,9):
      if i in b:
        rstring |= 1
      rstring <<= 1

    for i in range(1,9):
      if i in s:
        rstring |= 1
      rstring <<= 1

    rstring >>= 1
    return rstring

  def get_rstring(self):
    return format(self.rstring, 'b').zfill(16)

  def mutate(self, p):
    mask = 0
    for _ in range(16):
      if random.random() < p:
        mask |= 1
      mask <<= 1

    mask >>= 1
    self.rstring ^= mask

if __name__ == "__main__":
  r = Rulestring([2, 3], [2, 3, 4])
  print(r.get_rstring())
  r.mutate(0.5)
  print(r.get_rstring())

