------------------------------------------------------------------------------------------------------------------
This program just provided three different parallelizations of the SUMMA 
algorithm - I used it to experiment with how CUDA works, specifically its
blocks and threads. The three parallelizations were:
	1. Having each block represent every element in the N x N matrix.
	
	2. Having each block be a "window" in the N x N matrix that you accummulate
	the terms in when doing the matrix multiplty.
	
	3. Same as 1, except that each element were represented by threads.

------------------------------------------------------------------------------------------------------------------
Note: This program must be run on a CUDA machine.

To compile, please type,
	nvcc MatrixMultiplySquare.cu
into the command prompt.

To run the program, please type
	./a.out
into the command prompt.

