import csv
import time
from maze.evolve import Population
import matplotlib.pyplot as plt
from maze.maze_ca import MazeCA
from maze.media import make_files, clear_temp_folders


def run_experiment(epoch_n=30, pop_size=50, fit_ratio=0.2, elitism=0.2, mutation=0.05):
  start_time = time.time()
  avges = []
  maxes = []
  fails = []
  pop = Population(pop_size, fit_ratio, elitism, mutation)
  rules, ranks = None, None
  for epoch in range(epoch_n - 1):
    print("Epoch:", epoch + 1)
    avg, max, fail = pop.iterate()
    avges.append(avg)
    maxes.append(max)
    fails.append(fail)

  print("Epoch:", epoch_n)
  avg, max, fail, ranks = pop.select()
  avges.append(avg)
  maxes.append(max)
  fails.append(fail)
  epochs = [i for i in range(1, epoch_n + 1)]

  exp_n = f"fr_{fit_ratio}"
  for e in pop.inds[:3]:
    save_experiment(e.b, e.s, exp_n)

  dir = f'out/{exp_n}'
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([f"{e.b}_{e.s}" for e in pop.inds])
    write.writerow([e.get_rstring() for e in pop.inds])
    write.writerow(ranks)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(epochs, fails, color='red')
  plt.plot(epochs, avges, color='green')
  plt.plot(epochs, maxes, color='blue')
  ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
  plt.xlabel('Epochs', fontsize=14)
  plt.savefig(f'{dir}/fitness.png', bbox_inches='tight')

  print("--- %s seconds ---" % (time.time() - start_time))
  print(exp_n)
  return avges[-1]


def save_experiment(B, S, exp_n):
  B.sort()
  S.sort()

  # Rulestring is in B/S notation (Birth/Survival)
  rulestring = ''.join(str(i) for i in B) + '_' + ''.join(str(i) for i in S)
  print(f"Running CA {rulestring}")
  rulestring = f"{exp_n}/{rulestring}"

  ca = MazeCA(B, S)
  ca.generate(media=True)
  make_files(final_state=ca.X, name="generation", rstring=rulestring, clear=True)

  cells, regions, = ca.find_regions(media=True)
  make_files(final_state=ca.X, name="regions", rstring=rulestring, clear=False)

  success = ca.merge_regions(cells, regions, media=True)
  make_files(final_state=ca.X, name="merging", rstring=rulestring, clear=False)

  if success:
    ends, length = ca.dead_ends_and_path_length()
    print("Dead ends:", ends)
    print("Solution length:", length)
    # M = find_sol_path(M)
    # save_final_image(M, path=f'./out/{rulestring}/solution.png', ax=init_image())
  else:
    print("Region merge failed")
  clear_temp_folders()


if __name__ == "__main__":
  # epochs = [10, 30, 50]
  # pops = [20, 50]
  # elitisms = [0.1, 0.2, 0.5]
  # mutation = [0.01, 0.05, 0.1]
  # for fit_ratio in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
  #   best_fit = 0
  #   config = None, None, None, None

  run_experiment()
