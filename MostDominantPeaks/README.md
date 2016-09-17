------------------------------------------------------------------------------------------------------------------
This project was a parallelization of the most dominant peak finding algorithm
described in "A Computational Model for Periodic Pattern Perception Based On
Frieze and Wallpaper Groups" by Li et al. (2004), specifically the part where
the minimum distanes, Di, are computed for each peak. The parallelization is
as follows.

For a given set of points P1, P2, ..., PN, each block will compute the minimum
distances for a subset of these points. For each point inside the subset, each 
thread inside the block will compute the minimum distance for a subset of the 
points coming before that point. So for that single point in the block-subset, 
the threads will have a bunch of minimum distances corresponding to that point. 
These minimum distances will be reduced to a single value to represent the 
overall minimum distance of that point. This computation will be repeated
for all the points inside that block's subset, in parallel with respect to
the blocks. To see a more mathematical description of the parallelization,
please read 


------------------------------------------------------------------------------------------------------------------
Note: This program must be run on a CUDA machine.

To compile, please type,
	nvcc MostDominantPeaks.cu
into the command prompt.

To run the program, please type
	./a.out
into the command prompt.

