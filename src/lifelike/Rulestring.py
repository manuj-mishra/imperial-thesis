import random

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy import stats

from lifelike.CAs import CA, GRID_SIZE, MimicCA
from lifelike.constants import CHROMOSOME_LEN
from util import binary

class Rulestring:
  def __init__(self, rstring, b, s):
    self.rstring = rstring
    self.b = b
    self.s = s

  @classmethod
  def from_rstring(cls, rstring):
    b= binary.ones(rstring >> (CHROMOSOME_LEN // 2))
    s = binary.ones(rstring)
    return cls(rstring, b, s)

  @classmethod
  def from_bs(cls, b, s):
    bnum = 0
    for pos in b:
      bnum |= 1 << (CHROMOSOME_LEN - pos - 1)
    snum = 0
    for pos in s:
      snum |= 1 << ((CHROMOSOME_LEN // 2) - pos - 1)
    rstring = bnum + snum
    return cls(rstring, b, s)

  @classmethod
  def random_binary(cls):
    binarr = np.random.binomial(1, random.random(), size=CHROMOSOME_LEN)
    binarrstr = ''.join(binarr.astype(str))
    rstring = int(binarrstr, 2)
    b = np.where(binarr[:CHROMOSOME_LEN // 2] == 1)[0]
    s = np.where(binarr[CHROMOSOME_LEN // 2:] == 1)[0]
    return cls(rstring, b, s)

  @classmethod
  def random_decimal(cls):
    rstring = random.randint(1, 2 ** CHROMOSOME_LEN - 1)
    return Rulestring.from_rstring(rstring)

  def get_rstring(self):
    return format(self.rstring, 'b').zfill(CHROMOSOME_LEN)

  def set_rstring(self, rstring):
    self.rstring = rstring
    self.b = binary.ones(rstring >> (CHROMOSOME_LEN // 2))
    self.s = binary.ones(rstring)

  def mutate(self, p):
    mask = 0
    for _ in range(CHROMOSOME_LEN):
      if random.random() < p:
        mask |= 1
      mask <<= 1
    mask >>= 1
    self.set_rstring(self.rstring ^ mask)
    return self.rstring

  def loss(self, true, ics, hyperparams):
    losses = []
    for ic in ics:
      # pred = CA.random(self.b, self.s)
      pred = CA(ic, self.b, self.s)
      for step in range(hyperparams["eval_step"]):
          step_size = random.randint(1, hyperparams["max_step"])
          true_active = true.step_from(pred.X, step_size)
          pred_active = pred.step(step_size)
          losses.append(np.mean(pred.X ^ true.X))
          # losses.append(self.three_res_loss(pred.X, true.X))
          if not true_active and not pred_active:
            break
    return np.mean(losses)

  def three_res_loss(self, a, b):
    ksize = GRID_SIZE // 2
    kmid = np.ones((ksize, ksize)) / (ksize**2)
    high = np.mean(a ^ b)
    mid = np.mean(convolve2d(a, kmid, mode='valid').round().astype('bool') ^ convolve2d(b, kmid, mode='valid').round().astype('bool'))
    low = (a.sum() < a.size) ^ (b.sum() < b. size)
    return (low + mid + high) / 3
    # return mid

# if __name__ == "__main__":
#   # r = Rulestring.from_bs({1}, {})
#   # true = MimicCA.empty({3}, {2, 3})
#   # hyperparams = {"max_step": 1, "eval_step": 1}
#   # ics = [np.random.random((GRID_SIZE, GRID_SIZE)) > 0.5 for i in range(1)]
#   # print(r.loss(true, ics, hyperparams))
