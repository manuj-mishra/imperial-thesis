import random

import tensorflow as tf
from tensorflow import keras


class SlimeNetKeras(keras.Model):

  def __init__(self, inputs, outputs):
    super(SlimeNetKeras, self).__init__(inputs, outputs)
    self.nx, self.ny = 20, 20

  def train_step(self, data):
    state, solution = data
    state = state[0]
    solution = solution[0]
    with tf.GradientTape() as tape:
      n_steps = random.randint(1, 128)
      loss = 0
      for _ in range(n_steps):
        loss += self.compiled_loss(state[1], solution, regularization_losses=self.losses)
        inputs = []
        for i in range(self.ny):
          for j in range(self.nx):
            inputs.append(self.neumann_neighbourhood(state, i, j))
        print(len(inputs))
        delta = self(inputs, training=True)
        tf.transpose(delta)
        tf.reshape(delta, (16, self.ny, self.nx))
        snapshot = state[0]
        state += delta
        state[0] = snapshot
      loss += self.compiled_loss(state[1], solution, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(solution, state[1])
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

  def neumann_neighbourhood(self, data, cy, cx):
    res = [data[:, cy, cx]]
    res.append(data[:, 0 if cy == 0 else cy - 1, cx])
    res.append(data[:, 0 if cy == self.nx - 1 else cy + 1, cx])
    res.append(data[:, cy, 0 if cx == 0 else cx - 1])
    res.append(data[:, cy, 0 if cx == self.ny - 1 else cx + 1])
    tf.transpose(res)
    return res