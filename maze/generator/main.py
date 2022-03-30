import csv

from generator.evaluate import solution_length
from generator.generate_maze import generate_maze
from generator.media import make_files, clear_temp_folders, init_image
from generator.region_merge import get_regions, region_merge

B = [6, 7, 8]
S = [1, 6, 7, 8]


clear_temp_folders()
# Rulestring is in B/S notation (Birth/Survival)
rulestring = ''.join(str(i) for i in B) + '_' + ''.join(str(i) for i in S)

print("1. Generating maze ...")
ax = init_image()
X = generate_maze(B, S, folder='temp/gen_frames', ax=ax)
make_files(frame_folder='gen_frames', rstring=rulestring, name="generation", final_state=X, clear=True)

print("2. Finding regions ...")
ax = init_image()
cells, regions, M = get_regions(X, folder='temp/reg_frames', ax=ax)
make_files(frame_folder='reg_frames', rstring=rulestring, name="regions", final_state=M, clear=False)

print("3. Merging regions ...")
ax = init_image()
M, success= region_merge(regions, cells, M, folder='temp/merge_frames', ax=ax)
make_files(frame_folder='merge_frames', rstring=rulestring, name="merging", final_state=M, clear=False)

print("4. Evaluating maze ...")
if success:
  solution_length(M)
else:
  print("Region merge failed")
clear_temp_folders()

# file = open('sol_lens.csv', 'w+', newline ='')
# with file:
#     write = csv.writer(file)
#     write.writerow(sol_lens)