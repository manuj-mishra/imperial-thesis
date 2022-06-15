import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

from util.media import init_image, save_image, clear_temp_folders, make_files_clustered

GRID_SIZE = 51
RUN_ITERS = 5001
dt = 1.0  # Time delta
dA = 0.3
dB = dA / 2
lapl = np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
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
    A[int(GRID_SIZE / 2) - int(seed_size / 2):int(GRID_SIZE / 2) + int(seed_size / 2) + 1,
    int(GRID_SIZE / 2) - int(seed_size / 2):int(GRID_SIZE / 2) + int(seed_size / 2) + 1] \
      = np.random.normal(loc=0.5, scale=0.01, size=(seed_size, seed_size))
    B[int(GRID_SIZE / 2) - int(seed_size / 2):int(GRID_SIZE / 2) + int(seed_size / 2) + 1,
    int(GRID_SIZE / 2) - int(seed_size / 2):int(GRID_SIZE / 2) + int(seed_size / 2) + 1] \
      = np.random.normal(loc=0.25, scale=0.01, size=(seed_size, seed_size))
    return cls(f, k, A, B)

  @classmethod
  def splatter(cls, f, k, num_seeds=5):
    # Seed Size 3x3
    A = np.ones((GRID_SIZE, GRID_SIZE))
    B = np.zeros((GRID_SIZE, GRID_SIZE))
    indices = np.random.choice(np.arange(B.size), replace=False, size=num_seeds)
    coords = np.unravel_index(indices, B.shape)
    cx, cy = coords
    up = np.maximum(cy - 1, 0)
    down = np.minimum(cy + 1, GRID_SIZE - 1)
    left = np.maximum(cx - 1, 0)
    right = np.minimum(cx + 1, GRID_SIZE - 1)
    splat_coords = (np.concatenate((left, cx, right)), np.concatenate((up, cy, down)))
    B[splat_coords] = 1
    return cls(f, k, A, B)

  def step(self, steps=1):
    for _ in range(steps):
      A_new = self.A + (
          dA * convolve2d(self.A, lapl, mode='same', boundary='wrap', fillvalue=0)
          - (self.A * self.B * self.B)
          + (self.f * (1 - self.A))
      ) * dt
      B_new = self.B + (
          dB * convolve2d(self.B, lapl, mode='same', boundary='wrap', fillvalue=0)
          + (self.A * self.B * self.B)
          - ((self.f + self.k) * self.B)
      ) * dt

      if np.all(self.A == A_new) and np.all(self.B == B_new):
        return False

      self.A = np.copy(A_new)
      self.B = np.copy(B_new)

      if np.isnan(np.min(A_new)) or np.isnan(np.min(B_new)):
        return False

    return True

  def state(self):
    return self.B / (self.A + self.B)

  def run(self, rname, fname="", media=False):
    folder, fig, ax = None, None, None
    if media:
      folder = "temp/gen_frames"
      fig, ax = init_image()
    frames_per_image = 100

    for i in range(RUN_ITERS):
      self.step()

      if not i % frames_per_image and media:
        print(f"{i}/{RUN_ITERS}")
        save_image(self.state(), i, ax, cmap='Spectral', folder=folder)

    if media:
      make_files_clustered(final_state=self.state(), fname=f"gen_{fname}", rname=rname)
      # clear_temp_folders()
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
  ca = CA.splatter(f=0.03, k=0.06)
  ca.run(rname='test', media=True)
