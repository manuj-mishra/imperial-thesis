import random

import numpy as np

from lifelike.CAs import CA
from lifelike.constants import CHROMOSOME_LEN
from util import binary

# EVAL_ITERS = 10     # Number of CAs simulated per rulestring
EVAL_STEPS = 10     # Number of steps evaluated per CA
MAX_STEP_SIZE = 5   # Max size of a single CA step

class Rulestring:
  def __init__(self, rstring, b, s):
    self.rstring = rstring
    self.b = b
    self.s = s

  @classmethod
  def from_rstring(cls, rstring):
    b = binary.ones(rstring >> (CHROMOSOME_LEN // 2))
    s = binary.ones(rstring)
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
    rstring = random.randint(0, 2 ** CHROMOSOME_LEN - 1)
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
          step_size = random.randint(1, 10 * (step + 1))
          true_active = true.step_from(pred.X, step_size)
          pred_active = pred.step(step_size)
          losses.append(np.mean(pred.X ^ true.X))
          if not true_active and not pred_active:
            break
    return np.mean(losses)
