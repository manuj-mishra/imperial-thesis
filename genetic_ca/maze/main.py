import csv
import time
from maze.population import Population
import matplotlib.pyplot as plt
from maze.maze_ca import MazeCA
from maze.media import make_files, clear_temp_folders

EPOCH_N = 40


def run_experiment(path_len_bias, pop_size=50, elitism=0.5, mutation=0.05, epoch_n=EPOCH_N):
  start_time = time.time()
  mean_deads = []
  mean_paths = []
  pop = Population(pop_size, path_len_bias, elitism, mutation)
  for epoch in range(epoch_n - 1):
    print("Epoch:", epoch + 1)
    mean_dead_ends, mean_path_lens = pop.iterate()
    mean_deads.append(mean_dead_ends)
    mean_paths.append(mean_path_lens)

  print("Epoch:", epoch_n)
  mean_dead_ends, mean_path_lens, elite_scores = pop.select()
  mean_deads.append(mean_dead_ends)
  mean_paths.append(mean_path_lens)
  print(f"Training Time: {((time.time() - start_time) / epoch_n):.2f} s per epoch")
  save_experiments(pop, elite_scores, mean_paths, mean_deads)


def save_experiments(pop, elite_scores, mean_paths, mean_deads):
  exp_n = f"bias{pop.path_len_bias * 100}_pop{pop.pop_size}_el{pop.elitism * 100}_mut{pop.mutation * 100}"
  for rulestring in pop.inds[:3]:
    rulestring.b.sort()
    rulestring.s.sort()

    rname = ''.join(str(i) for i in rulestring.b) + '_' + ''.join(str(i) for i in rulestring.s)
    rname = f"{exp_n}/{rname}"

    save_experiment(rulestring, rname)

  dir = f'out/{exp_n}'
  file = open(f'{dir}/elite.csv', 'w+', newline='')
  with file:
    write = csv.writer(file)
    write.writerow([f"{e.b}_{e.s}" for e in pop.inds])
    write.writerow([e.get_rstring() for e in pop.inds])
    write.writerow(elite_scores)

  epochs = [i for i in range(1, EPOCH_N + 1)]
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_ylim(bottom=0)
  plt.plot(epochs, mean_paths, color='green')
  plt.plot(epochs, mean_deads, color='red')
  ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
  plt.xlabel('Epochs', fontsize=14)
  plt.savefig(f'{dir}/fitness.png', bbox_inches='tight')


def save_experiment(rulestring, rname, attempt=1):
  if attempt == 0:
    print(f"Running CA {rname}")

  ca = MazeCA(rulestring.b, rulestring.s)
  ca.generate(media=True)
  make_files(final_state=ca.X, fname="generation", rname=rname, clear=True)

  cells, regions, = ca.find_regions(media=True)
  make_files(final_state=ca.X, fname="regions", rname=rname)

  success = ca.merge_regions(cells, regions, media=True)
  make_files(final_state=ca.X, fname="merging", rname=rname)

  if success:
    ends, length, reachable = ca.metrics(media=True)
    make_files(final_state=ca.X, fname="evaluation", rname=rname)
    print("Dead ends:", ends)
    print("Solution length:", length)
    print("Reachable:", reachable)
    # M = find_sol_path(M)
    # save_final_image(M, path=f'./out/{rulestring}/solution.png', ax=init_image())
  else:
    print(f"Attempt {attempt}: Region merge failed")
    if attempt < 3:
      print("Trying again")
      save_experiment(rulestring, rname, attempt=attempt + 1)

  clear_temp_folders()


if __name__ == "__main__":
  for pop_size in [20, 50]:
    for elitism in [0.2, 0.5, 0.8]:
      for mutation in [0.025, 0.05, 0.1]:
        run_experiment(0)
        run_experiment(0.5)
        run_experiment(1)
