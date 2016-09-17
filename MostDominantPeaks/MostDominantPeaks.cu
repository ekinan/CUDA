#include <stdio.h>
#include <cfloat>

#define BLOCKS 4 // Each block is a peak(s) whose minimum distance we're trying to calculate
#define THREADS 16 // Each thread is a preceding point(s) of the peak in its block

#define MAX 100 // Max value of point coordinate
#define N 50 // Number of points to be considered

/*
Basic "Point" class used as a representative example to test the algorithm
*/
struct Point {
	double x; // x coordinate
	double y; // y coordinate
	double d; // minimum distance

	__host__ __device__ Point(double x_, double y_) : x(x_), y(y_), d(DBL_MAX) {
	}

	/*
	Points are displayed as ((x,y),d) tuples, where the first componeont is the
	point itself, second component is its minimum distance. Note if d = DBL_MAX,
	"dMAX" is displayed instead for d to avoid outputting a very large number
	*/
	__host__ __device__ void DisplayPoint() const {
		printf("((%0.0f,%0.0f),", x, y);
		if (d == DBL_MAX) {
			printf("dMAX");
		} else {
			printf("%0f", d);
		}
		printf(")");
	}
};

/*
Displays all of the points in an array of points
*/
__host__ __device__ void DisplayPoints(Point* pts, int size); 

/*
Creates a test case of points having size N. The x and y coordinates of the point
are randomly generated. Returns a pointer to this test case
*/
__host__ Point* GetTestCase();

/*
Creates a device copy of the array of points stored in "pts." Returns a pointer
to this copy
*/
__host__ Point* GetDeviceCopy(Point* pts);


/*
Calculates the Euclidian distance squared between p1 and p2.
*/
__device__ double Distance(const Point &p1, const Point &p2);

/*
Extracts the minimum distance corresponding to the set of points
under the current thread's jurisdiction. pts is the array of points
whose distances we need, blockPoint corresponds to the current
point whose minimum distance we're calculating.
*/
__device__ double ExtractMinimumDistance(Point* pts, int blockPoint);

/*
Does a pair-wise reduction of the array in minArray. After execution,
minArray should hold a single value representing the minimum distance
for the current peak under consideration.

Note: Make sure the number of threads are a power of 2.
*/
__device__ void ReduceToSingleMinimum(double *minArray);

/*
For a given array of points P[0...n-1], where size = n,
this function calculates their minimum distances.

Each block is a point Pi, while the threads calculate Pis
distance from Pk, where 0 <= k < i, 1 <= i < n, and store
the minimum distance they calculate in minArray[threadIdx.x].
If a thread has more than one Pk, then minArray[threadIdx.x]
is the minimum distance of all of its Pks relative to Pi.

These minimums are subsequently reduced to a single minimum value,
where this value is Pis minimum distance
*/
__global__ void CalculateMinimumDistances(Point* pts, int size);


int main(int argc, char* argv[]) {
	Point* pts = GetTestCase();
	Point* dev_pts = GetDeviceCopy(pts);

	printf("Before Minimum Distance:\n");	
	DisplayPoints(pts, N);
	printf("\n");

	CalculateMinimumDistances<<<BLOCKS, THREADS>>>(dev_pts, N);
	cudaMemcpy(pts, dev_pts, N*sizeof(Point), cudaMemcpyDeviceToHost);

	printf("After Minimum Distance:\n");
	DisplayPoints(pts, N);
	printf("\n");

	free(pts); cudaFree(dev_pts);

	return 0;
}


__host__ __device__ void DisplayPoints(Point* pts, int size) {
	for (int i = 0; i < size; ++i) {
		pts[i].DisplayPoint();
		printf(" ");
	}
	printf("\n");
}

__host__ Point* GetTestCase() {
	srand(time(0)); // Reset the random seed each time.
	Point* pts = (Point*)malloc(N*sizeof(Point));
	for (int i = 0; i < N; ++i) {
		pts[i] = Point(rand()%MAX, rand()%MAX);		
	}

	return pts;
}

__host__ Point* GetDeviceCopy(Point* pts) {
	Point* dev_pts;
	cudaMalloc((void**)&dev_pts, N*sizeof(Point));
	cudaMemcpy(dev_pts, pts, N*sizeof(Point), cudaMemcpyHostToDevice);

	return dev_pts;
}




__device__ double Distance(const Point &p1, const Point &p2) {
	double xdiff = p1.x-p2.x;
	double ydiff = p1.y-p2.y;
	return xdiff*xdiff+ydiff*ydiff; // return (x1-x2)^2 + (y1-y2)^2
}

__device__ double ExtractMinimumDistance(Point* pts, int blockPoint) {
	double dmin = DBL_MAX; // Start minimum distance off as infinity
	for (int i = threadIdx.x; i < blockPoint; i += blockDim.x) { // For each Pk under the thread's jurisdiction
		double dist = Distance(pts[i], pts[blockPoint]); // Calculate Pks distance from Pi, where Pi = pts[blockPoint]
		if (dist < dmin) { // Update this thread's minimum distance if dmin is smaller than dist
			dmin = dist;
		}
	}

	return dmin;
}

__device__ void ReduceToSingleMinimum(double *minArray) {
	int numPairs = THREADS/2; //Number of thread pairs
	while (numPairs) { // Reduce until numPairs = 0, indicating a single element in minArray
		if (threadIdx.x < numPairs) { 
			int start = 2*threadIdx.x; // Pairs are separated by two elements
			// Assign the smaller value of the two pair elements to this thread
			minArray[threadIdx.x] = minArray[start] <= minArray[start+1] ? minArray[start] : minArray[start+1];	
		}
		numPairs /= 2; // Halve the pairs for the next iteration
		__syncthreads(); // Wait until all threads have processed their reduction
	}
}

__global__ void CalculateMinimumDistances(Point* pts, int size) {
	__shared__ double minArray[THREADS]; // Shared array that stores the minimum distances of each thread
	for (int blockPoint = blockIdx.x+1; blockPoint < size; blockPoint += gridDim.x) { // For all points Pi in block's jurisdiction
		minArray[threadIdx.x] = ExtractMinimumDistance(pts, blockPoint); // Get the thread minimum distances
		__syncthreads(); // Wait until all threads have finished
		ReduceToSingleMinimum(minArray); // Now reduce to a single minimum to get Pis minimum distance.
		pts[blockPoint].d = minArray[0];
	}
}
