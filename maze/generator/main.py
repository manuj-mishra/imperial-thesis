from generator.generate_maze import generate_maze
from generator.media import make_files, clear_temp_folders
from generator.region_merge import get_regions, region_merge

B = [2, 3]
S = [2, 3, 4]

# Rulestring is in B/S notation (Birth/Survival)
rulestring = ''.join(str(i) for i in B) + '_' + ''.join(str(i) for i in S)

clear_temp_folders()

print("1. Generating maze ...")
X = generate_maze(B, S, folder='gen_frames')
make_files(frame_folder='gen_frames', rstring=rulestring, name="generation", final_state=X, clear=True)

print("2. Finding regions ...")
cells, regions, M, n = get_regions(X, folder='reg_frames')
make_files(frame_folder='reg_frames', rstring=rulestring, name="regions", final_state=M, clear=False)

print("3. Merging regions ...")
M = region_merge(regions, cells, M, n, folder='merge_frames')
make_files(frame_folder='merge_frames', rstring=rulestring, name="merging", final_state=M, clear=False)

# print("4. Evaluating maze ...")
# evaluate(M)

clear_temp_folders()
