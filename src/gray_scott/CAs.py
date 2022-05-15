import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

from util.media import init_image, save_image, clear_temp_folders, make_files_clustered

GRID_SIZE = 97
RUN_ITERS = 10000
dt = 1.0  # Time delta
dA = 1.0
dB = 0.5
lapl = np.array([[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]])
eps = 0.00001


class CA:
  def __init__(self, f, k, A, B):
    self.f = f
    self.k = k
    self.A = A
    self.B = B

  @classmethod
  def patch(cls, f, k, seed_size=5):
    A = np.ones((GRID_SIZE, GRID_SIZE))
    B = np.zeros((GRID_SIZE, GRID_SIZE))
    B[int(GRID_SIZE / 2) - int(seed_size / 2):int(GRID_SIZE / 2) + int(seed_size / 2) + 1,
    int(GRID_SIZE / 2) - int(seed_size / 2):int(GRID_SIZE / 2) + int(seed_size / 2) + 1] = np.ones(
      (seed_size, seed_size))
    return cls(f, k, A, B)

  def step(self, steps=1):
    for _ in range(steps):
      A_new = self.A + (
          dA * convolve2d(self.A, lapl, mode='same', boundary='fill', fillvalue=0)
          - (self.A * self.B * self.B)
          + (self.f * (1 - self.A))
      ) * dt
      B_new = self.B + (
          dB * convolve2d(self.B, lapl, mode='same', boundary='fill', fillvalue=0)
          + (self.A * self.B * self.B)
          - (self.k * self.B)
      ) * dt

      if np.allclose(self.A, A_new) and np.allclose(self.B, B_new):
        return True

      self.A = np.copy(A_new)
      self.B = np.copy(B_new)

    return False

  def state(self):
    return self.B / (self.A + self.B)

  def run(self, rname, fname="", media=False):
    folder, fig, ax = None, None, None
    if media:
      folder = "temp/gen_frames"
      fig, ax = init_image()
    frames_per_image = 100

    for i in range(RUN_ITERS):
      has_converged = self.step()

      if has_converged:
        if media:
          save_image(self.state(), i, ax, cmap='Spectral', folder=folder)
        break

      if not i % frames_per_image and media:
        save_image(self.state(), i, ax, cmap='Spectral', folder=folder)

    if media:
      make_files_clustered(final_state=self.state(), fname=f"gen_{fname}", rname=rname)
      clear_temp_folders()
      plt.close(fig)
    return self.state()


class MimicCA(CA):
  @classmethod
  def empty(cls, f, k):
    A = np.empty((GRID_SIZE, GRID_SIZE))
    B = np.empty((GRID_SIZE, GRID_SIZE))
    return cls(f, k, A, B)

  def step_from(self, A, B, steps=1):
    self.A = A
    self.B = B
    return self.step(steps)

if __name__ == "__main__":
  # ca = CA.patch(f=0.055, k=0.117) # Slow growing circle
  # ca = CA.patch(f=0.17398293947208127, k=0.34699284376436873)
  ca = CA.patch(f=0.045, k=0.099)
  ca.run(rname='tets', media=True)


