import numpy as np

from gray_scott.CAs import CA

NUM_STEPS = 10
STEP_SIZE = 500


class Chromosome:
  def __init__(self, state, control):
    self.state = state
    self.control = control

  @classmethod
  def threshold(cls, control):
    f = np.random.uniform(low=0.0, high=0.25)
    k = (np.sqrt(f) / 2) - f
    k = max(np.random.normal(loc=k, scale=0.1), 0)
    return cls(state=np.array([f, k]), control=control)

  @classmethod
  def random(cls, control):
    f = np.random.uniform(low=0.0, high=0.30)
    k = np.random.uniform(low=0.0, high=0.08)
    return cls(state=np.array([f, k]), control=control)

  def loss(self, real):
    losses = []
    pred = CA.splatter(self.state[0], self.state[1])
    for _ in range(NUM_STEPS):
      true_active = real.step_from(pred.A, pred.B, STEP_SIZE)
      pred_active = pred.step(STEP_SIZE)
      if np.isnan(np.min(real.state())):
        break

      if np.isnan(np.min(pred.state())):
        break

      losses.append(np.mean(np.abs(pred.state() - real.state()) / real.state()))
      if not true_active and not pred_active:
        break
    return np.mean(losses) if losses else 1
