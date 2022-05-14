import numpy as np

from gray_scott_simple.CAs import CA

NUM_STEPS = 5
STEP_SIZE = 1000

class Chromosome:
  def __init__(self, state, control):
    self.state = state
    self.control = control

  @classmethod
  def near_threshold(cls, control):
    f = np.random.uniform(low=0.0, high=0.0625)
    k = -f + 0.5 * np.sqrt(f)
    return cls(state=np.array([f, k]), control=control)

  @classmethod
  def random(cls, control):
    f = np.random.uniform(low=0.0, high=0.0625)
    k = np.random.uniform(low=0.0, high=0.0625)
    return cls(state=np.array([f, k]), control=control)

  def loss(self, true):
    losses = []
    pred = CA.patch(self.state[0], self.state[1])
    for _ in range(NUM_STEPS):
        true_active = true.step_from(pred.A, pred.B, STEP_SIZE)
        pred_active = pred.step(STEP_SIZE)
        if np.isnan(np.min(true.state())):
          break

        if np.isnan(np.min(pred.state())):
          break

        losses.append(np.mean(np.abs(pred.state() - true.state())))
        if not true_active and not pred_active:
          break
    return np.mean(losses) if losses else 1
