### TODO

#### Small
- Make it so that the temp folder is automatically created and deleted
- ~~Remove maze/gifs folder~~
- Combine b_chrom and s_chrom into one dictionary

#### Medium
- ~~Media creation for dead_ends~~ and path_length
- ~~Find bug with dead_ends in maze~~
- Test all functions
- ~~Refactor maze like slime (i.e. MazeCA class which is called by Rulestring in evaluate)~~
- List all hyperparameters in slime and maze and try to integrate into chromosome
- Stop training based on EMA

#### Big
1) Add sobel filter based coefficients to slime chromosome
2) Grid search for hyperparameters on maze
3) Profiling, what code is taking ages to run
4) Count # of species over time

### Key
- 0 = space
- 1 = wall
- 2 = start / visited
- 3 = fringe
- 4 = goal
