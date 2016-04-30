/** *******************************************************************
 *  File Name : MemoryManager.h
 *  Define objects of quadtree.
 *  Define query polygons.
 *  Preallocate memory for all quadtree objects on CPU.
 *  Memory allocation on the GPU.
 ** *******************************************************************/
/**<************************# Includes ********************************/
#include<stdlib.h>
#include<stdio.h>
#include <stdbool.h>

/**<************************ Structure definition *********************/
typedef int POINTID;
typedef int NODEID;
typedef int BUFFID;
typedef int POLYID;

/**
 * The CPU has three main structures, NODE (nodes of the quadtree),
 * POINT (the input points for the quadtree) and
 * POINT_BUFFER (the array of points which each leaf node of the quadtree holds).
 */
typedef struct POINT {
	int x;
	int y;
	int index;
} POINT;

typedef struct POINT_BUFFER {
	// Array of points
	POINTID pointArray;
	unsigned int pointCount;
	// use a linked list to point to another NODE at same level.
	BUFFID nextId;
} POINT_BUFFER;

typedef struct NODE {
	// Level
	unsigned int level;
	// Keep track of type of NODE : LEAF, LINK
	unsigned int type;
	// Location in 2D space
	int posX;
	int posY;
	// Size of quadrant
	int width;
	int height;
	// Description of points
	int count[4];
	int total;
	int index;
	int parent_index;
	int offset;
	// For Adaptive implementation
	int open;
	NODEID child[4];
	// This handles the 4 regions 0,1,2,3 representing sub-quadrants
	BUFFID pBuffer;
	bool P0, P1, P2, P3;
} NODE;

// Definition of query polygon
typedef struct Polygon {
	// Size of Polygon.
	//xmin, ymin, xmax and ymax defines 4 corners of a polygon
	int xmin;
	int ymin;
	int width;
	int height;
	int xmax;
	int ymax;
	int count;
	int index;
	POLYID *PolyPointId;
} Polygon;

// Structure to store points from completely overlapping nodes in GPU.
typedef struct CP{
	int polygonIndexCP;
	int CPcount;
	NODEID nodeNumCP[64];
}CP;

// Structure to store points from partially overlapping nodes in GPU
typedef struct NODEPP{
	int xminPP;
	int yminPP;
	int xmaxPP;
	int ymaxPP;
}NODEPP;

typedef struct PP{
	int polygonIndexPP;
	int PPcount;
	NODEPP nodePP_array[64];
}PP;

/**<************************ Memory preallocation **********************/
// Preallocating memory for point structure.
typedef struct{
	POINT* pArray;
	int currCount,maxSize;
}point_array_t;
point_array_t pArr;

void allocatePointMemory(){
	pArr.pArray = (POINT*)malloc(100000000*sizeof(POINT));
	pArr.currCount = 0;
	pArr.maxSize = 1000000000;
	if (pArr.pArray != NULL) {
	}
}

// Function to fetch memory for point from preallocated memory.
POINT* getPoint(int id){
	if(id >= pArr.currCount || id < 0){
		exit(-1);
	}
	return &pArr.pArray[id];
}

// Allocate memory for a new point.
POINTID getNewPoint(){
	if (pArr.currCount >= pArr.maxSize){
		pArr.pArray = (POINT*)realloc(pArr.pArray, pArr.maxSize + 10000);
		pArr.maxSize+=10000;
	}
	return pArr.currCount++;
}

// Fetch memory for array of points.
POINTID getPointArray(int nelems){
	int strtId = pArr.currCount;
	pArr.currCount+=nelems;
	if (pArr.currCount >= pArr.maxSize){
		pArr.maxSize = (pArr.maxSize + 10000) > pArr.currCount ? pArr.maxSize + 10000:pArr.currCount + 1000;
		pArr.pArray = (POINT*)realloc(pArr.pArray, (pArr.maxSize)*sizeof(POINT));
	}
	return strtId;
}

// Preallocating memory for node structure.
typedef struct{
	NODE* nodeArray;
	int currCount,maxSize;
}node_array_t;
node_array_t nodeArr;

void allocateNodeMemory(){
	nodeArr.nodeArray = (NODE*)malloc(1000*sizeof(NODE));
	nodeArr.currCount = 0;
	nodeArr.maxSize = 1000;
}

// Function to fetch memory for point from preallocated memory.
NODE* getNode(int id){
	if(id >= nodeArr.currCount || id < 0){
		exit(-1);
	}
	return &nodeArr.nodeArray[id];
}

// Allocate memory for a new node.
NODEID getNewNode(){
	if (nodeArr.currCount >= nodeArr.maxSize){
		nodeArr.nodeArray = (NODE*)realloc(nodeArr.nodeArray, nodeArr.maxSize + 10000);
		nodeArr.maxSize+=10000;
	}
	return nodeArr.currCount++;
}

// Preallocating memory for point buffer structure.
typedef struct{
	POINT_BUFFER* bufferArray;
	int currCount,maxSize;
}buff_array_t;
buff_array_t bufferArr;

void allocatePOINTBUFFERMemory(){
	bufferArr.bufferArray = (POINT_BUFFER*)malloc(4000000*sizeof(POINT_BUFFER));
	bufferArr.currCount = 0;
	bufferArr.maxSize = 4000000;
}

// Function to fetch memory for point buffer from preallocated memory.
POINT_BUFFER* getBUFFER(int id){
	if(id >= bufferArr.currCount || id < 0){
		exit(-1);
	}
	return &bufferArr.bufferArray[id];
}

// Allocate memory for a new buffer.
BUFFID getNewBUFFER(){
	if (bufferArr.currCount >= bufferArr.maxSize){
		bufferArr.bufferArray = (POINT_BUFFER*)realloc(bufferArr.bufferArray, bufferArr.maxSize + 10000);
		bufferArr.maxSize+=10000;
	}
	return bufferArr.currCount++;
}

// Get the node indices of the level 3 nodes.
typedef struct {
	NODEID *Level3Nodes;
	int count;
}level3Node;
level3Node Level3NodeArray;

void levelThreeNode()
{
	Level3NodeArray.Level3Nodes = (NODEID*)malloc(64*sizeof(NODEID));
	int current = getNewNode() - 1 ;
	int i;
	int j = 0;
	for (i = 0; i<=current; i++){
		if(getNode(i)->level == 2){
			Level3NodeArray.count++;
			Level3NodeArray.Level3Nodes[j] = i;
			j++;
		}
	}
}

/**<****************** CUDA device memory allocation *****************/
// Allocate CUDA memory for Point structure.
POINT *d_POINT;
void allocatePointMemoryCuda(){
	cudaMalloc((void**)&d_POINT, sizeof(POINT)*pArr.currCount);
}

// Allocate CUDA memory for Point buffer structure.
POINT_BUFFER *d_POINT_BUFFER;
void allocatePoint_BufferMemoryCuda(){
	cudaMalloc((void**)&d_POINT_BUFFER, sizeof(POINT_BUFFER)*bufferArr.currCount);
}

// Allocate CUDA memory for Node structure.
NODE *d_NODE;
void allocateNodeMemoryCuda(){
	cudaMalloc((void**)&d_NODE, sizeof(NODE)*nodeArr.currCount);
}

//Allocate memory for level 3 nodes.
NODEID *d_NODE_In;
void allocateNodeIDMemoryCuda(){
	cudaMalloc((void**)&d_NODE_In, sizeof(NODEID)*Level3NodeArray.count);
}

/**<******************* Memory transfer from CPU to GPU **************/
// Copy Point structure to GPU.
void PointCudaCopy(){
	cudaMemcpy(d_POINT, pArr.pArray, sizeof(POINT)*pArr.currCount, cudaMemcpyHostToDevice);
}

// Copy Point buffer structure to GPU.
void Point_BufferCudaCopy(){
	cudaMemcpy(d_POINT_BUFFER, bufferArr.bufferArray, sizeof(POINT_BUFFER)*bufferArr.currCount, cudaMemcpyHostToDevice);
}

// Copy Node structure to GPU.
void NodeCudaCopy(){
	cudaMemcpy(d_NODE, nodeArr.nodeArray, sizeof(NODE)*nodeArr.currCount, cudaMemcpyHostToDevice);
}

//Copy level 3 nodes to GPU.
void NodeIDCudaCopy(){
	cudaMemcpy(d_NODE_In, Level3NodeArray.Level3Nodes, sizeof(NODEID)*Level3NodeArray.count, cudaMemcpyHostToDevice);
}


