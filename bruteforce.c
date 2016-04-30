/** *******************************************************************
 *  File name : bruteForce.cu
 *  Create random points and polygons.
 *  Perform a brute force search to find points in polygon on GPU.
 ** *******************************************************************/
/**<************************# Includes **********************/
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/time.h>
#include <stdbool.h>
#include "hw4_4A_timing.h"
#include<cstdlib>
#include <cuda.h>

/**<************************# Defines **************************/
#define __host__
#define __shared__
#define BUILD_FULL 1
#define BUILD_ADAPTIVE 2
#define MODE_RANDOM 1
#define MODE_FILE 2
#define TRUE 1
#define FALSE 0
#ifndef RANGE
#define RANGE 1024
#endif
#ifndef SIZE
#define SIZE 20000
#endif

/**<************************ Global variables **************/
int pointMode = MODE_RANDOM;
char *inputPointFileName;
char *outputTreeFileName;
int rangeSize = RANGE;
int bucketSize = 32;
int numPoints = SIZE;
int numPolygon =50;
int pointRangeX = RANGE;
int pointRangeY = RANGE;
typedef int POLYID;

/**<************************ Structure definition ************/
// Input data point
typedef struct POINT {
	int x;
	int y;
	int index;
} POINT;

// Point coordinates.
typedef struct NODEPP{
	int xminPP;
	int yminPP;
}NODEPP;

// Query polygon
typedef struct Polygon {
	// coordinates of corners of a polygon and its width and height.
	int xmin;
	int ymin;
	int width;
	int height;
	int xmax;
	int ymax;
	int count;
	int index;
	NODEPP nodePoint[20000];
} Polygon;

/**<***************** Performance functions ***************/
typedef struct perf_struct {
	double sample;
	struct timeval tv;
} perf;

perf timeRecord;

void init_perf(perf *t) {
	t->sample = 0;
}

// Start time measurement.
void start_perf_measurement(perf *t) {
	gettimeofday(&(t->tv), NULL);
}

// Stop time measurement.
void stop_perf_measurement(perf *t) {
	struct timeval ctv;
	gettimeofday(&ctv, NULL);
	struct timeval diff;
	timersub(&ctv, &(t->tv), &diff);
	double time = diff.tv_sec + diff.tv_usec / 1000000.0;
	t->sample = time;
}

// Print time difference.
void print_perf_measurement(perf *t) {
	double total_time = 0.0;
	double stdev = 0.0;
	printf("%f", t->sample);
}

/**<************************ Random number generator *********************/
/**
 * Generate random numbers.
 * 1- Generate random data points.
 * 2- Generate random numbers for polygon corners.
 ** **********************************************************************/

// Generate a random number within a range.
int randn(int range) {
	int a;
	a = rand() % range;
	return a;
}

// Create random query Polygons
Polygon *createRandomPolygon(unsigned int nPolygon, unsigned int range) {
	Polygon *polyarray = (Polygon *) malloc(sizeof(Polygon) * nPolygon);
	unsigned int index;
	for (index = 0; index < nPolygon; index++) {
		polyarray[index].xmin = randn(512);
		polyarray[index].ymin = randn(512);
		polyarray[index].width = 400;
		polyarray[index].height = 300;
		polyarray[index].xmax = polyarray[index].xmin + polyarray[index].width;
		polyarray[index].ymax = polyarray[index].ymin + polyarray[index].height;
		polyarray[index].index = index;
	}
	return polyarray;
}

// Create random data points.
POINT *createRandomPoints(unsigned int nPoints, unsigned int range) {
	POINT *pointArray = (POINT *) malloc(sizeof(POINT) * nPoints);
	unsigned int index;
	for (index = 0; index < nPoints; index++) {
		pointArray[index].x = randn(range);
		pointArray[index].y = randn(range);
		pointArray[index].index = index;
	}
	return pointArray;
}

POINT *createPoints(int numPoints) {
	POINT *pointArray;
	pointArray = createRandomPoints(numPoints, rangeSize);
	return pointArray;
}

/**<************************ Kernel function ********************/
/**<************************ Brute force search *********************/
/**
 * Function to search points inside polygon using brute force search method.
 * 1- Assign each thread to a data point.
 * 2- The thread checks if the point is within a given polygon.
 * 3- Repeat till all polygons are checked.
 * 4- Store the attributes of the polygon which satisfy the query.
 * 5- Then grid stride is done, so that the thread loops over to get the next point.
 ** *******************************************************************/

// Brute force search
__global__ void bruteForce(POINT* d_pointArray, Polygon* d_Polygon, int numPoints, int numPolygon)
{
	// Get global thread index.
	int tidG = threadIdx.x + blockIdx.x*blockDim.x;
	int pid;
	// Boundary check for threads.
	if(tidG < numPoints)
	{
		d_Polygon[pid].count = 0;
		// Loop till all points are checked.
		for(int tid = tidG; tid < numPoints; tid = tid + blockDim.x*gridDim.x)
		{
			// Loop till all polygons are checked
			for(pid = 0; pid < numPolygon; pid++)
			{
				// Check every point with every polygon.
				if ((d_pointArray[tid].x >= d_Polygon[pid].xmin) && (d_pointArray[tid].x <= d_Polygon[pid].xmax) && (d_pointArray[tid].y >= d_Polygon[pid].ymin) && (d_pointArray[tid].y <= d_Polygon[pid].ymax))
				{
					// Store the points that lie within a polygon.
					d_Polygon[pid].nodePoint[d_Polygon[pid].count].xminPP = d_pointArray[tid].x;
					d_Polygon[pid].nodePoint[d_Polygon[pid].count].yminPP = d_pointArray[tid].y;
					d_Polygon[pid].count++;
				}
			}
		}
	}
}

/**<************************ Main function ***************************/
/**
 * Main function.
 * 1- Create points randomly.
 * 2- Create polygons randomly.
 * 3- Copy points and polygons from host to device.
 * 4- Launch CUDA kernel to perform the search.
 * 5- Time the kernel execution.
 ** *******************************************************************/
int main(int argc, char **argv) {
	// Host variables
	int i;
	float diff_GPU;
	struct Polygon *randomPoly;
	int index;
	int j,p;
	float time = 0.0;
	int NumberofPolygons;
	// Device variables
	POINT *d_pointArray;
	Polygon *d_Polygon;
	// Create data points and polygons randomly.
	POINT *pointArray = createPoints(numPoints);
	randomPoly = createRandomPolygon(numPolygon, rangeSize);
	// Allocate device memory.
	cudaMalloc((void**)&d_pointArray, sizeof(POINT)*numPoints);
	cudaMalloc((void**)&d_Polygon, sizeof(Polygon)*numPolygon);
	// Memory transfer from host to device
	cudaMemcpy(d_pointArray, pointArray, sizeof(POINT)*numPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Polygon, randomPoly, sizeof(Polygon)*numPolygon, cudaMemcpyHostToDevice);
	// Time brute force search.
	perfTimer timeRecord;
	initTimer(&timeRecord);
	cudaEvent_t start_GPU, stop_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	cudaEventRecord(start_GPU, 0);
	// Kernel launch
	bruteForce<<<(numPoints/1024), 1024>>>(d_pointArray, d_Polygon, numPoints, numPolygon);
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != CUDA_SUCCESS){
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
	}
	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&diff_GPU, start_GPU, stop_GPU);
	time = diff_GPU;
	// Calculate number of polygons per second
	NumberofPolygons = numPolygon / time;
	printf("%d\n", NumberofPolygons);
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);
	return 0;
}



