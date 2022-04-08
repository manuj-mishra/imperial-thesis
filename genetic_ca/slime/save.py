import csv
import time

from maze.evaluate import dead_ends_and_path_length, find_sol_path
from maze.generate_maze import generate_maze, get_regions, region_merge
from maze.media import make_files, clear_temp_folders, init_image, save_final_image

B = [3, 2]
S = [7, 5, 3, 2, 1]


def save(B, S, exp_n):
  B.sort()
  S.sort()

  # Rulestring is in B/S notation (Birth/Survival)
  rulestring = ''.join(str(i) for i in B) + '_' + ''.join(str(i) for i in S)
  print(f"Running CA {rulestring}")
  rulestring = f"{exp_n}/{rulestring}"

  ax = init_image()
  X = generate_maze(B, S, media=True, folder='temp/gen_frames', ax=ax)
  make_files(frame_folder='gen_frames', rstring=rulestring, name="generation", final_state=X, clear=True)

  ax = init_image()
  cells, regions, M = get_regions(X, media=True, folder='temp/reg_frames', ax=ax)
  make_files(frame_folder='reg_frames', rstring=rulestring, name="regions", final_state=M, clear=False)

  ax = init_image()
  M, success = region_merge(regions, cells, M, media=True, folder='temp/merge_frames', ax=ax)
  make_files(frame_folder='merge_frames', rstring=rulestring, name="merging", final_state=M, clear=False)

  if success:
    _, ends, length = dead_ends_and_path_length(M)
    print("Solution length:", length)
    print("Dead ends:", ends)
    # M = find_sol_path(M)
    # save_final_image(M, path=f'./out/{rulestring}/solution.png', ax=init_image())
  else:
    print("Region merge failed")
  clear_temp_folders()

if __name__ == "__main__":
  save(B, S, '.')
