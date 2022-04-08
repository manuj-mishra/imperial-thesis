import csv
import time

from slime.evolve import Population
import matplotlib.pyplot as plt
from slime.media import init_image, make_files, clear_temp_folders
from slime.slime_ca import SlimeCA


def run_experiment(epoch_n=50, pop_size=30, fit_ratio=0.9, elitism=0.2, mutation=0.05):
  start_time = time.time()

  avges = []
  maxes = []
  pop = Population(pop_size, fit_ratio, elitism, mutation)
  rules, ranks = None, None
  for epoch in range(epoch_n - 1):
    print("Epoch:", epoch + 1)
    avg, max = pop.iterate()
    avges.append(avg)
    maxes.append(max)

  print("Epoch:", epoch_n)
  avg, max, ranks = pop.select()
  avges.append(avg)
  maxes.append(max)
  epochs = [i for i in range(1, epoch_n + 1)]

  exp_n = f"fr_{fit_ratio}"
  for e in pop.inds[:3]:
    save_experiment(e, exp_n)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(epochs, avges, color='green')
  plt.plot(epochs, maxes, color='blue')
  ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
  plt.xlabel('Epochs', fontsize=14)
  plt.savefig(f'{dir}/fitness.png', bbox_inches='tight')
  plt.show()
  print("--- %s seconds ---" % (time.time() - start_time))
  return avges[-1]

def save_experiment(chromosome, exp_n):

  rulestring = str(chromosome)
  print(f"Running CA {rulestring}")
  rulestring = f"{exp_n}/{rulestring}"

  ca = SlimeCA(chromosome)
  ax = init_image()
  X = ca.run(media=True, folder='temp/gen_frames', ax=ax)
  make_files(frame_folder='gen_frames', rstring=rulestring, name="generation", final_state=X, clear=True)
  print("SLIME SIZE:", ca.slime_size())
  print("FOOD REACHED:", ca.food_reached())
  clear_temp_folders()

if __name__ == "__main__":
  run_experiment()
