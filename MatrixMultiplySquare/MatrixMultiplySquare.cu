#include <stdio.h>

#define N 10 //Rows and columns of the matrix

#define SmallCols 2 // The col size of the tiny window
#define SmallRows 2 // The row size of the tiny window

#define ThreadCols 5 // The col size of a single window
#define ThreadRows 5 // The row size of a single window

/*
This was the code I used to run a matrix multiply in CUDA.
I used the idea of the SUMMA algorithm, where the psuedocode
for a serialized version is as follows:

	C[N][N]; // Initialize the C matrix to 0
	for (int k = 0; k < N; ++k) {
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}

There were three versions of the matrix mutiply that I decided
to test out, mostly to gain familiarity with how to access
a particular element of the matrix given the grid size
and the distribution of threads in each block:
	1) SingleBlock: Here, each block represented a single element
	   of the N x N matrix.

	2) SmallBlockMultiply: Here, each block had only one thread
	   and the size of the block matrix was smaller than the original
	   matrix.

	3) ThreadBlockMatrixMultiply: Here, each block had TreadRows x ThreadCols
	   submatrices each and they represented a single element in the overall
	   matrix (the number of blocks was determined through division).
*/

/*
Returns M[i][j], since I used an array to store the matrix values.
*/
__host__ __device__ int GetMatrixEntry(int *M, int i, int j);

/*
Returns the index of the linear array corresponding to row i and column j.
*/
__host__ __device__ int GetLinearIndex(int i, int j);


/*
Displays the matrixM with an optional description attached.
*/
void DisplayMatrix(const char* descp, int *M);

/*
Generates the Iota matrix 1 to N and returns a pointer to it.
*/
int* GenerateIotaMatrix();

/*
Generates the zero matrix and returns a pointer to it.
*/
int* GenerateZeroMatrix();

/*
Returns a pointer to a device copy of the matrix M
*/
int* CreateDeviceMatrixCopy(int *M);

/*
Copies the contents of the device matrix, dev_M, to
the host matrix M.
*/
void CopyMatrixDeviceToHost(int *M, int *dev_M);

/*
Kernels for the three versions of matrix multiply I tried out (see long comment
above).

Note that k corresponds to the "k" value outlined in the outer loop in the
psuedocode above.
*/
__global__ void SingleBlockMatrixMultiply(int *A, int *B, int *C, int k);
__global__ void SmallBlockMatrixMultiply(int *A, int *B, int *C, int k);
__global__ void ThreadBlockMatrixMultiply(int *A, int *B, int *C, int k);


/*
Executes the memory allocation to host and device parts of the CUDA process
(to avoid clutting the calling function with repetitive code)
*/
void ExecuteInitialSteps(int* *A, int* *B, int* *C, int* *dev_A, int* *dev_B, int* *dev_C);

/*
Displays matrices A, B and C (to avoid cluttering with repetitive code)
*/
void DisplayMatrices(int *A, int *B, int *C);

/*
Frees all of of the memories. A, B and C are host matrices,
while their "dev_" prefixed versions are their device versions.
*/
void FreeMemories(int *A, int *B, int *C, int *dev_A, int *dev_B, int *dev_C);

/*
These functions run the procedures that end up calling their corresponding kernel functions.
*/
void RunSingleBlockMatrixMult();
void RunSmallBlockMatrixMult();
void RunThreadBlockMatrixMult();

int main(int argc, char* argv[]) {
	//RunSingleBlockMatrixMult();
	//RunSmallBlockMatrixMult();
	RunThreadBlockMatrixMult();

	return 0;
}


//Kernel functions

__global__ void SingleBlockMatrixMultiply(int *A, int *B, int *C, int k) {
	int i = blockIdx.y;
	int j = blockIdx.x;

	// Each block is a single element of the matrix
	C[GetLinearIndex(i, j)] += A[GetLinearIndex(i,k)]*B[GetLinearIndex(k,j)];
}

__global__ void SmallBlockMatrixMultiply(int *A, int *B, int *C, int k) {
	int i = blockIdx.y;
	int j = blockIdx.x;

	while (i < N && j < N) { // Keep multiplying while we are within the main matrices' bounds
		C[GetLinearIndex(i,j)] += A[GetLinearIndex(i,k)]*B[GetLinearIndex(k,j)];

		int temp_i = i+SmallRows; //Move the window down
		while (temp_i < N) {
			C[GetLinearIndex(temp_i,j)] += A[GetLinearIndex(temp_i,k)]*B[GetLinearIndex(k,j)];
			temp_i += SmallRows;
		}

		int temp_j = j+SmallCols; //Move the window to the right
		while (temp_j < N) {
			C[GetLinearIndex(i,temp_j)] += A[GetLinearIndex(i,k)]*B[GetLinearIndex(k,temp_j)];
			temp_j += SmallCols;
		}

		i += SmallRows; //Now move the window down along the diagonal
		j += SmallCols;	
	}
}


__global__ void ThreadBlockMatrixMultiply(int *A, int *B, int *C, int k) {	
	int i = blockIdx.y*blockDim.y+threadIdx.y;
	int j = blockIdx.x*blockDim.x+threadIdx.x;

	// Each thread represents a single element of the matrix, the threads
	// dispersed along the blocks
	if (i < N && j < N) {
		C[GetLinearIndex(i, j)] += A[GetLinearIndex(i,k)]*B[GetLinearIndex(k,j)];
	}
}

void ExecuteInitialSteps(int* *A, int* *B, int* *C, int* *dev_A, int* *dev_B, int* *dev_C) {
	*A = GenerateIotaMatrix(); // A and B are the iota matrix
	*B = GenerateIotaMatrix();

	*C = GenerateZeroMatrix(); // C starts off as zero

	*dev_A = CreateDeviceMatrixCopy(*A);
	*dev_B = CreateDeviceMatrixCopy(*B);

	*dev_C = CreateDeviceMatrixCopy(*C);
}

void DisplayMatrices(int *A, int *B, int *C) {
	DisplayMatrix("A:", A); printf("\n");
	DisplayMatrix("B:", B); printf("\n");
	DisplayMatrix("C:", C); printf("\n");
}


void FreeMemories(int *A, int *B, int *C, int *dev_A, int *dev_B, int *dev_C) {
	free(A); free(B); free(C);
	cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
}






void RunSingleBlockMatrixMult() {
	int *A, *B, *C;
	int *dev_A, *dev_B, *dev_C;

	ExecuteInitialSteps(&A, &B, &C, &dev_A, &dev_B, &dev_C);

	dim3 grid(N, N); // Here, each block is a single element of the matrix
	for (int k = 0; k < N; ++k) {
		SingleBlockMatrixMultiply<<<grid, 1>>>(dev_A, dev_B, dev_C, k);
	}

	CopyMatrixDeviceToHost(C, dev_C);

	DisplayMatrices(A, B, C);
	FreeMemories(A, B, C, dev_A, dev_B, dev_C);
}

void RunSmallBlockMatrixMult() {
	int *A, *B, *C;
	int *dev_A, *dev_B, *dev_C;

	ExecuteInitialSteps(&A, &B, &C, &dev_A, &dev_B, &dev_C);


	dim3 grid(SmallCols,SmallRows); // Here, we use a tiny sub matrix that traverses along the entire matrix
	for (int k = 0; k < N; ++k) {
		SmallBlockMatrixMultiply<<<grid, 1>>>(dev_A, dev_B, dev_C, k);
	}

	CopyMatrixDeviceToHost(C, dev_C);

	DisplayMatrices(A, B, C);
	FreeMemories(A, B, C, dev_A, dev_B, dev_C);
}

void RunThreadBlockMatrixMult() {
	int *A, *B, *C;
	int *dev_A, *dev_B, *dev_C;

	ExecuteInitialSteps(&A, &B, &C, &dev_A, &dev_B, &dev_C);


	dim3 grid((N+ThreadCols-1)/ThreadCols,(N+ThreadRows-1)/ThreadRows); // The number of blocks needed to capture the entire matrix
	dim3 threads(ThreadCols, ThreadRows); // The size of the submatrix per block
	for (int k = 0; k < N; ++k) {
		ThreadBlockMatrixMultiply<<<grid, threads>>>(dev_A, dev_B, dev_C, k);
	}

	CopyMatrixDeviceToHost(C, dev_C);

	DisplayMatrices(A, B, C);
	FreeMemories(A, B, C, dev_A, dev_B, dev_C);
}





__host__ __device__ int GetMatrixEntry(int *M, int i, int j) {
	return *(M+(i*N+j));
}

__host__ __device__ int GetLinearIndex(int i, int j) {
	return (N*i+j);	
}

void DisplayMatrix(const char* descp, int *M) {
	printf("%s\n", descp);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%d ", GetMatrixEntry(M, i, j));	
		}
		printf("\n");
	}
}

int* GenerateIotaMatrix() {
	int* M = (int*)malloc(N*N*sizeof(int));
	for (int i = 0; i < N*N; ++i) {
		M[i] = i+1;
	}

	return M;
}

int* GenerateZeroMatrix() {
	int* M = (int*)malloc(N*N*sizeof(int));
	for (int i = 0; i < N*N; ++i) {
		M[i] = 0;
	}

	return M;
}

int* CreateDeviceMatrixCopy(int *M) {
	int *dev_M;
	cudaMalloc((void**)&dev_M, N*N*sizeof(int));
	cudaMemcpy(dev_M, M, N*N*sizeof(int), cudaMemcpyHostToDevice);

	return dev_M;
}

void CopyMatrixDeviceToHost(int *M, int *dev_M) {
	cudaMemcpy(M, dev_M, N*N*sizeof(int), cudaMemcpyDeviceToHost);
}
