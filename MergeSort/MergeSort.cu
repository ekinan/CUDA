#include <stdio.h>

/*
Generic swap function in C, I got this off of stack overflow.
*/
#define swap(x,y) do { \
		unsigned char swap_temp[sizeof(x) == sizeof(y) ? (signed)sizeof(x) : -1]; \
		memcpy(swap_temp, &y, sizeof(x)); \
		memcpy(&y, &x, sizeof(x)); \
		memcpy(&x, swap_temp, sizeof(x)); \
	} while(0)


#define N 41
#define MAX 100

#define BLOCKS 4 // Number of blocks/subproblems that we're solving for the overall array.
#define THREADS 10 // Number of subproblems inside a single block that we're solving.


/*
Creates and returns a pointer to an array sorted in decreasing order. If size = N,
then the returned array is A[0..N-1] with A[0] = N, A[1] = N-1, ..., A[N-1] = 1
*/
int* CreateUnsortedArray(int size);

/*
This creates and returns a pointer to a randomly generated array.
*/
int* CreateRandomArray(int size);

/*
Implementation of insertion sort. I copied and pasted this online. The idea is that
each sub-array inside a block will be sorted by insertion sort. Sorts the array's elements
in the range [first, last]
*/
__host__ __device__ void InsertionSort(int* array, int first, int last);

/*
Prints the elements of an array with a description of them attached in the range [first, last]
*/
__host__ __device__ void PrintArray(const char* descp, int* array, int first, int last);


/*
Merges the elements of the two sub-arrays corresponding to indices [leftFirst, leftLast] and
[rightFirst, rightLast], respectively into a single, sorted array [leftFirst, rightLast].
Note that rightFirst = leftLast + 1 for this function to work. The merge code was also
obtained online.
*/
__host__ __device__ void Merge(int *array, int *temp, int leftFirst, int leftLast, int rightFirst, int rightLast);


/*
This function does only the "merging" part of the "MergeSort" function below.
It is intended to be called after each block in the array has been sorted,
so that this would do a pair-wise merge of the blocks into a single array.
*/
__global__ void DoMergeOnly(int *array, int *temp, int size);

/*
Sorts the array using a method that behaves like merge sort.
Note that temp is passed on to avoid dynamically allocating
memory on the GPU.

The idea here is that the array is paritioned into blocks,
each block holds a single sub array. Each block is then
partitioned into threads, where the threads themselves are sub-arrays
of the block.

The thread arrays are first sorted using insertion sort (since they're
the smallest "unit" of the array). Then, each pair of adjacent
threads are merged together until a single, merged array remains;
this array is the current block sorted in ascending order.

After the blocks each contain sorted subarrays. the function "DoMergeOnly"
is called on a single block with the same number of threads as there were
blocks in the preceding step. It merges the sub arrays in the blocks together
and the final array is then sorted in ascending order.

Note: If the number of blocks does not evenly divide the array size,
then the remaining elements are added on to the last block. Same with
the threads for the size of the array in each block.
*/
__global__ void MergeSort(int *array, int *temp, int size);

int main(int argc, char* argv[]) {

	// Create the test array
	int* a = CreateUnsortedArray(N);
	int *dev_a, *dev_temp;
	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);

	// Allocate the temporary array (to avoid doing it on the GPU)
	cudaMalloc((void**)&dev_temp, N*sizeof(int));

	// Print the array's contents
	PrintArray("Before MergeSort:", a, 0, N-1);
	printf("\n");

	// Do the first step of merge sort, where the array contains BLOCKS
	// number of sorted subarrays after this statement
	MergeSort<<<BLOCKS, THREADS>>>(dev_a, dev_temp, N);

	// Output the intermediate step (to verify that the sub arrays are sorted
	cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);
	PrintArray("Intermediate:", a, 0, N-1);
	printf("\n");

 	// Now do a pair-wise merge on the subarrays so that the final array is sorted.
	DoMergeOnly<<<1, BLOCKS>>>(dev_a, dev_temp, N);

	// Output the final array
	cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost);
	PrintArray("After MergeSort:", a, 0, N-1);
	printf("\n");


	// Free the memory
	free(a); cudaFree(dev_a); cudaFree(dev_temp);

	return 0;
}


int* CreateUnsortedArray(int size) {
	int* array =(int*)malloc(size*sizeof(int));
	for (int i = 0; i < size; ++i) {
		array[i] = size-i;
	}

	return array;
}

int* CreateRandomArray(int size) {
	int* array = (int*)malloc(size*sizeof(int));
	for (int i = 0; i < size; ++i) {
		array[i] = rand() % MAX;
	}

	return array;
}



__host__ __device__ void InsertionSort(int* array, int first, int last) {
	for (int i = first; i <= last; ++i) {
		int j = i;
		while (j > first && array[j] < array[j-1]) {
			swap(array[j], array[j-1]);
			--j;
		}
	}
}

__host__ __device__ void PrintArray(const char* descp, int* array, int first, int last) {
	printf("%s\n", descp);
	for (int i = first; i <= last; ++i) {
		printf("%d ", array[i]);
	}
	printf("\n");
}

__host__ __device__ void Merge(int *array, int *temp, int leftFirst, int leftLast, int rightFirst, int rightLast) {
	int i, j, k;
	i = leftFirst;
	k = leftFirst;
	j = rightFirst;

	while (i <= leftLast && j <= rightLast) {
		if (array[i] < array[j]) {
			temp[k++] = array[i++];
		} else {
			temp[k++] = array[j++];
		}
	}

	while (i <= leftLast) {
		temp[k++] = array[i++];
	}

	while (j <= rightLast) {
		temp[k++] = array[j++];
	}

	for (i = leftFirst; i <= rightLast; ++i) {
		array[i] = temp[i];
	}
}

__global__ void MergeSort(int *array, int *temp, int size) {
	int elemPerBlock = size/gridDim.x; // Number of array elements per sub array of the block
	int blockFirst = blockIdx.x*elemPerBlock; // The starting point of the array for this block

	// Ending point of the sub array. Note if we're at the last block, we simply set this
	// To be N - 1, where N = size
	int blockLast = (blockIdx.x == (gridDim.x - 1) ? size - 1 : blockFirst + elemPerBlock - 1); 

	// Number of array elements in the sub array for a single thread
	int elemPerThread = elemPerBlock/blockDim.x;
	int threadFirst = blockFirst + threadIdx.x*elemPerThread; // Same logic as blockFirst, save for threads

	// Same logic as blockLast, save now we're doing it for threads
	int threadLast = (threadIdx.x == (blockDim.x - 1) ? blockLast : threadFirst + elemPerThread - 1);

	InsertionSort(array, threadFirst, threadLast); // Sort the subarrays in each thread by insertion sort
	__syncthreads(); // Wait until all threads are finished

	//Now we merge pair by pair
	int numThreads = (blockDim.x+2-1)/2; // Initial pair is the number of threads per block over 2
	while (numThreads > 1) { // The greater than 1 is because we may not have our threads as a power of 2, so we take the ceiling but ceiling never reaches 0
		if (threadIdx.x < numThreads) { // Is a valid pair that we are considering
			int startId = threadIdx.x*2;
			threadFirst = blockFirst + startId*elemPerThread; // Start of the first pair
			int splitPoint = threadFirst+elemPerThread; // End location of the first pair

			if (threadIdx.x == (numThreads - 1)) { // If it's the last thread, we want the end to be the last
				threadLast = blockLast;
			} else { // Otherwise, we traverse 2*M elements, where M is the number of elements per pair to the end
				threadLast = threadFirst+2*elemPerThread-1;
			}
			Merge(array, temp, threadFirst, splitPoint-1, splitPoint, threadLast);
		}
		__syncthreads(); // Wait until all threads are done
		numThreads = (numThreads+2-1)/2;
		elemPerThread *= 2;
	}


	if (threadIdx.x == 0) { //Finish off the merge. We did not set test condition to 0 above because we always took the ceiling. So we address single thread case here.
		int splitPoint = blockFirst+elemPerThread; // Calculate the split point

		//Last merge
		Merge(array, temp, blockFirst, splitPoint-1, splitPoint, blockLast);
	}
}

// This code is the same as the latter steps in the above. I did not write this
// in a separate function at the time because I wanted to test my code first
// and avoid repeating the first 5-6 lines of computation in the threads.
__global__ void DoMergeOnly(int *array, int *temp, int size) {
	int elemPerBlock = size;
	int blockFirst = 0;
	int blockLast = size-1;

	int elemPerThread = elemPerBlock/blockDim.x;

	//Now we merge pair by pair
	int numThreads = (blockDim.x+2-1)/2;
	while (numThreads > 1) {
		if (threadIdx.x < numThreads) {
			int startId = threadIdx.x*2;
			int threadFirst = blockFirst + startId*elemPerThread;
			int splitPoint = threadFirst+elemPerThread;

			int threadLast;
			if (threadIdx.x == (numThreads - 1)) {
				threadLast = blockLast;
			} else {
				threadLast = threadFirst+2*elemPerThread-1;
			}

			Merge(array, temp, threadFirst, splitPoint-1, splitPoint, threadLast);
		}
		__syncthreads();
		numThreads = (numThreads+2-1)/2;
		elemPerThread *= 2;
	}

	if (threadIdx.x == 0) { //Finish off the merge
		int splitPoint = blockFirst+elemPerThread;
		Merge(array, temp, blockFirst, splitPoint-1, splitPoint, blockLast);
	}
}
