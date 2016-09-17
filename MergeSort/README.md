------------------------------------------------------------------------------------------------------------------
This program was a parallelization of merge sort that I worked on to see how an
existing algorithm could be parallelized. The idea was to let each block
represent a sub-array, and each thread inside the block is a sub-array of
the sub-array. The thread sub-arrays are sorted using a basic sorting algorithm
(here insertion sort), and they are then merged together to form a sorted
sub-array. The sorted block sub-arrays are then merged again to form a single,
sorted array. Please see the comments starting at line 55 of the code for more
details.

Note that I used an online implementation of insertion sort and the merging
part of merge sort because the focus of the CUDA code was the parallelization.
I've done both algorithms before in my CS 163 class.

------------------------------------------------------------------------------------------------------------------
Note: This program must be run on a CUDA machine.

To compile, please type,
	nvcc MergeSort.cu
into the command prompt.

To run the program, please type
	./a.out
into the command prompt.

