/** *******************************************************************
 *  File name : quadtreeGPU.cu
 *  Construct quadtree in CPU.
 *  Traverse quadtree in GPU to find PIP.
 ** *******************************************************************/
/**<************************# Includes ********************************/
#include<stdio.h>
#include<stdlib.h>
#include"MemoryManager.h"
#include<unistd.h>
#include<sys/time.h>
#include <stdbool.h>
#include<stdlib.h>
#include<cstdlib>
#include <cuda.h>
#include <math.h>
#ifdef __CDT_PARSER__

/**<************************# Defines *********************************/
#define __host__
#define __shared__
#define CUDA_KERNEL_DIM(...)
#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>
#endif
#define BUILD_FULL 1
#define BUILD_ADAPTIVE 2
#define MODE_RANDOM 1
#define MODE_FILE 2
#define TRUE 1
#define FALSE 0
#define pMax 32
#ifndef RANGE
#define RANGE 1024
#endif

/**<***************** Global variables ****************************/
int pointMode = MODE_RANDOM;
char *inputPointFileName;
char *outputTreeFileName;
int rangeSize = RANGE;
int bucketSize = 32;
int numPoints = 3000;
int numLevels = 3;
int numSearches = 0;
int printTree = 0;
int outputTree = 0;
int quadTreeMode = BUILD_FULL;
int numPolygon = 1099120;
int pointRangeX = RANGE;
int pointRangeY = RANGE;
int completeIndex = 0;
int NotIndex = 0;
int PartialIndex = 0;
int arraysize = 100;

/**<***************** enums ******************************/
enum {
	TYPE_NONE = 0, TYPE_ROOT, TYPE_LINK, TYPE_LEAF
};

enum {
	FullyOverlapped = 0, PartiallyOverlapped
};

/**<****************** Generic Functions ****************/
/**<****************** Timing Functions *****************/
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
}

/**<************************ Random number generator *********************/
/**
 * Generate random numbers.
 * 1- Generate random data points.
 * 2- Generate random numbers for polygon corners.
 ** **********************************************************************/
int randn(int range) {
	int a;
	a = rand() % range;
	return a;
}

int randnRng(int N, int M)
{
	int r;
	// Initialize random seed
	srand (time(NULL));
	// Generate number between a N and M
	r = M + random() / (RAND_MAX / (N - M + 1) + 1);
	return r;
}

// Creates random polygon with random xmin, ymin, width and height.
Polygon *createRandomPolygon(unsigned int nPolygon, unsigned int range) {
	Polygon *polyarray = (Polygon *) malloc(sizeof(Polygon) * nPolygon);
	int PointArraySize = getNewPoint() -1 ;
	unsigned int index;
	for (index = 0; index < nPolygon; index++) {
		polyarray[index].xmin = randn(723);
		polyarray[index].ymin = randn(723);
		polyarray[index].width = randnRng(100, 300);
		polyarray[index].height = randnRng(100, 300);
		polyarray[index].xmax = polyarray[index].xmin + polyarray[index].width;
		polyarray[index].ymax = polyarray[index].ymin + polyarray[index].height;
		polyarray[index].count = 0;
		polyarray[index].index = index;
	}
	return polyarray;
}

// Generate random data points, given a number of points and its range.
POINTID createRandomPoints(unsigned int nPoints, unsigned int range) {
	POINTID pointArray = getPointArray(nPoints);
	unsigned int index;
	for (index = 0; index < nPoints; index++) {
		POINT *p=getPoint(pointArray+index);
		p->x=randn(range);
		p->y=randn(range);
		p->index=index;
	}
	return pointArray;
}

/**<************************ Tree Functions ***************************/
/**<************************ Set node *********************************/
/**
 * Set node parameters.
 * 1- Set appropriate values for x, y, width, height and level of a node.
 * 2- Initialize rest of the node parameters.
 ** ********************************************************************/
void setNode(NODEID nodeid, int x, int y, int w, int h, int type, int level) {
	// Get memory for node.
	NODE* node=getNode(nodeid);
	// Set the 5 parameters.
	node->posX = x;
	node->posY = y;
	node->width = w;
	node->height = h;
	node->level = level;
	// Reset all of the tracking values.
	int i;
	for (i = 0; i < 4; i++)
	{
		node->child[i] = -1;
		node->count[i] = 0;
	}
	node->total = 0;
	node->index = 0;
	node->offset = 0;
	node->open = TRUE;
	node->type = type;
	node->pBuffer = -1;
}

/**<************** Count number of nodes ******************/
int countNodesQuadTree(NODEID nodeid) {
	int sum = 0;
	int i;
	if(nodeid == -1)
		return 0;
	// Depth first traversal to find the total number of nodes.
	if (getNode(nodeid)->type == TYPE_LEAF) {
		return 1;
	} else {
		for (i = 0; i < 4; i++) {
			sum = sum + countNodesQuadTree(getNode(nodeid)->child[i]);
		}
	}
	return sum + 1;
}

/**<*************** Assign index and offset ********************/
/**
 * 1- DFS traversal of tree to assign indices.
 * 2- Count data points in leaf node.
 ** ************************************************************/
void walkNodeTable(NODE *parent, NODEID nodeid, int *offset, int *index) {
	NODE* node=getNode(nodeid);
	BUFFID ptrID;
	int i;
	if (node == NULL)
		return;
	if (parent)
		node->parent_index = parent->index;
	else
		node->parent_index = -1;
	// Assign index and offset.
	node->index = *index;
	node->offset = *offset;
	// Advance the next index.
	*index = *index + 1;
	if (node->type == TYPE_LEAF) {
		int count = 0;
		// Get indices of points
		for (ptrID = node->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId){
			POINT_BUFFER *ptr= getBUFFER(ptrID);
			count = count + ptr->pointCount;
		}
		// Assign total number of points in node.
		node->total = count;
		*offset = *offset + count;
	} else {
		for (i = 0; i < 4; i++) {
			walkNodeTable(node, node->child[i], offset, index);
		}
	}
}

/**<************************ Build quadtree ****************************/
/**<************************ Create new point buffer ********************/
/**
 * Create new point buffer.
 * 1- Get memory for new buffer from preallocated memory.
 * 2- Allocate memory inside each buffer to hold 32 data points.
 * 3- Assign first point to buffer and set point count.
 * 4- Initialize pointer to next buffer.
 ** *********************************************************************/
BUFFID newPointBuffer(POINT *p) {
	// Allocate memory for point buffer.
	BUFFID ptrID = getNewBUFFER();
	POINT_BUFFER *ptr= getBUFFER(ptrID);
	// Allocate a bucket for 32 data points.
	(ptr->pointArray) = getPointArray(bucketSize);
	// Get point and set parameters.
	*(getPoint(ptr->pointArray)) = *p;
	ptr->pointCount = 1;
	ptr->nextId = -1;
	return ptrID;
}

/**<************************ Assign point to a node ********************/
/**
 * Add a point to leaf node.
 * 1- If a buffer of a leaf node is not full, add points to that buffer.
 * 2- If the buffer is full, then add new buffer to end of list.
 ** *********************************************************************/
void addPointToNode(NODE *node, POINT *p) {
	BUFFID ptrID;
	BUFFID lastBufferID;
	// Add points till buffer is full.
	if (node->pBuffer != -1) {
		for ( ptrID = node->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId) {
			POINT_BUFFER *ptr= getBUFFER(ptrID);
			// Add to the end of the list.
			if (ptr->pointCount < bucketSize) {
				*(getPoint((ptr->pointArray)+(ptr->pointCount))) = *p;
				ptr->pointCount++;
			}
			BUFFID lastBufferID = ptrID;
		}
		// Boundary case of adding a new link.
		getBUFFER(lastBufferID)->nextId = newPointBuffer(p);
	} else {
		node->pBuffer = newPointBuffer(p);
	}
}

/**<****************** Get direction of node ***************/
/**
 * 1- Get node direction based on input data point.
 * 2- A point is checked to see if it lies in NW, NE, SW or SE direction of a node.
 * 3- Return direction to calling function.
 ** *********************************************************/
int getNodeDirection(NODE *nParent, struct POINT *p) {
	int posX, posY;
	int x, y;
	int index;
	// Get the point.
	x = p->x;
	y = p->y;
	// Child width and height
	int width = nParent->width / 2;
	int height = nParent->height / 2;
	// Decide direction (North west (NW), North east (NE), South west (SW), South east (SE) of a point).
	for (index = 0; index < 4; index++) {
		switch (index) {
		case 0: // NW
			posX = nParent->posX;
			posY = nParent->posY + height;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return 0;
			}
			break;
		case 1: // NE
			posX = nParent->posX + width;
			posY = nParent->posY + height;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return 1;
			}
			break;
		case 2: // SW
			posX = nParent->posX;
			posY = nParent->posY;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return 2;
			}
			break;
		case 3: // SE
			posX = nParent->posX + width;
			posY = nParent->posY;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return 3;
			}
			break;
		}
	}

	exit(-1);
	return (-1);
}

/**<******************* Build node *****************************/
/**
 * 1- Check the direction of the node.
 * 2- Calculate and assign the node's x, y, width and height based on direction.
 * 3- Assign the node level.
 ** *************************************************************/
void buildNode(NODEID node, NODE *nParent, int direction) {
	int posX, posY, width, height, level;
	switch (direction) {
	case 0: // NW
		posX = nParent->posX; //0
		posY = nParent->posY + nParent->height / 2; //512
		break;
	case 1: // NE
		posX = nParent->posX + nParent->width / 2;
		posY = nParent->posY + nParent->height / 2;
		break;
	case 2: // SW
		posX = nParent->posX;
		posY = nParent->posY;
		break;
	case 3: // SE
		posX = nParent->posX + nParent->width / 2;
		posY = nParent->posY;
		break;
	}
	// Width and height of the child node is simply 1/2 parent.
	width = nParent->width / 2;
	height = nParent->height / 2;
	// Set the level.
	level = nParent->level + 1;
	setNode(node, posX, posY, width, height, TYPE_NONE, level);
}
NODEID createChildNode(NODE *parent, int direction) {
	NODEID node = getNewNode();
	buildNode(node, parent, direction);
	return (node);
}

/**<************************ Build full quadtree *************************/
/**
 * 1- Get node direction.
 * 2- Build node in that direction.
 * 3- Repeat till bottom of tree is reached.
 * 4- If bottom of the tree is reached, then add point to leaf node's buffer.
 ** **********************************************************************/
void buildFullTree(NODEID nodeid, unsigned int level, POINT *p) {
	NODEID dirNode;
	NODEID child;
	int direction;
	NODE*node=getNode(nodeid);
	// Check if bottom of tree is reached.
	if (node->level == level) {
		addPointToNode(node, p);
	} else {
		// Get direction of point in the node.
		direction = getNodeDirection(node, p);
		dirNode = node->child[direction];
		if (dirNode!=-1) {
			buildFullTree(dirNode, level, p);
		} else {
			// Create child node in that direction.
			child = createChildNode(node, direction);
			node->child[direction] = child;
			// Assign node type.
			if (getNode(child)->level == level)
				getNode(child)->type = TYPE_LEAF;
			else
				getNode(child)->type = TYPE_LINK;
			buildFullTree(node->child[direction], level, p);
		}
	}
}

/**<************************ Build adaptive quadtree ********************/
/**
 * Quadtree grows based on input points.
 * 1- Get direction of the node based on the input point.
 * 2- Build node in that direction.
 * 3- Add points to the node's buffer till predetermined limit is reached.
 * 4- Once limit is reached, divide node to sub nodes.
 * 5- Build child nodes and push points to it and repeat.
 ** *********************************************************************/
void buildAdaptiveTree(NODEID node, unsigned int level, POINT *p);

void pushPointToChildren(NODE *node, unsigned int level, POINT *p) {
	NODEID dirNode;
	NODEID child;
	int direction;
	// Get node direction.
	direction = getNodeDirection(node, p);
	dirNode = node->child[direction];
	if (dirNode) {
		buildAdaptiveTree(dirNode, level, p);
	} else {
		// Build node in that direction.
		child = createChildNode(node, direction);
		node->child[direction] = child;
		getNode(child)->type = TYPE_LEAF;
		buildAdaptiveTree(node->child[direction], level, p);
	}
}

void pushAllNodePointsToChildren(NODE *node, unsigned int level) {
	BUFFID ptrID;
	int link = 0;
	int i;
	ptrID = node->pBuffer;
	if (ptrID == -1)
		return;
	POINT_BUFFER *ptr= getBUFFER(ptrID);
	// Should have only 1 bucket's worth.
	if (ptr->nextId != -1) {
		printf("pushAllNodePointsToChildren: error\n");
		exit(-1);
	}
	// Get direction of node and push points to the node's buffer.
	for (i = 0; i < ptr->pointCount; i++) {
		pushPointToChildren(node, level, (getPoint((ptr->pointArray)+i)));
	}
	node->pBuffer = -1;
	node->open = FALSE;
	if (node->type == TYPE_LEAF)
		node->type = TYPE_LINK;
}

void buildAdaptiveTree(NODEID nodeid, unsigned int level, POINT *p) {
	NODEID dirNode;
	NODEID child;
	int direction;
	NODE*node=getNode(nodeid);
	// Have we reached the bottom : force to put point there : linked buckets
	if (getNode(nodeid)->level == level) {
		addPointToNode(node, p);
		return;
	}
	if (node->open == FALSE) {
		// If got to here, then this is an empty link node and we push point down.
		pushPointToChildren(node, level, p);
		return;
	}
	// All of the following checks assume node is open.
	// Node is open but point buffer is empty.
	if (node->pBuffer == -1) {
		addPointToNode(node, p);
		return;
	}
	// Check if the point belongs on this node.
	if ((node->pBuffer != -1) && (getBUFFER(node->pBuffer)->pointCount < bucketSize)) {
		// Add to current pointBuffer.
		POINT_BUFFER *ptr = getBUFFER(node->pBuffer);
		POINT* pt = getPoint(ptr->pointArray+ptr->pointCount);
		pt = p;
		getBUFFER(node->pBuffer)->pointCount++;
		return;
	}
	if ((node->pBuffer != -1) && (getBUFFER(node->pBuffer)->pointCount == bucketSize)) {
		// Full Buffer
		pushPointToChildren(node, level, p);
		// Push all points and delete this node's buffer.
		pushAllNodePointsToChildren(node, level);
		return;
	}
	printf("Should never get here \n");
	exit(-1);
}

/**<************************ Build quadtree ************************/
void buildQuadTree(NODEID node, unsigned int level, POINTID pointArray,
		int nPoints, int treeType) {
	int i;
	for (i = 0; i < nPoints; i++) {
		if (treeType == BUILD_FULL)
			buildFullTree(node, level, getPoint(pointArray+i));
		else
			buildAdaptiveTree(node, level, getPoint(pointArray+i));
	}
}


/**<************************ Print functions **************************/
/**
 * Functions to print quadtree.
 * 1- DFS traversal of quadtree.
 * 2- Print quadtree node details to a file.
 * 3- Print data points in leaf node to a file.
 * ** *******************************************************************/
void printTableNode(FILE *fp, NODEID nodeid) {
	int i;
	if(nodeid==-1)
		return;
	NODE*node=getNode(nodeid);
	// Print node details of tree to file.
	fprintf(fp, "%d %d : [%d %d] %d %d : %d ", node->index, node->offset,
			node->posX, node->posY, node->width, node->height, node->total);
	fprintf(fp, "next line \n");
	fprintf(fp, " %d :", node->parent_index);
	for (i = 0; i < 4; i++) {
		int index = -1;
		if (node->child[i]!=-1) {
			index = getNode(node->child[i])->index;
		}
		fprintf(fp, " %d", index);
	}
	fprintf(fp, "\n");
}

// Print node details to file.
void printTableNodeDataFile(FILE *fp, NODEID nodeid) {
	int i;
	if (nodeid == -1)
		return;
	printTableNode(fp, nodeid);
	for (i = 0; i < 4; i++) {
		printTableNodeDataFile(fp, getNode(nodeid)->child[i]);
	}
}

// Print index of data point.
void printLeafPointsDataFile(FILE *fp, NODEID nodeid) {
	BUFFID ptrID;
	int i;
	if(nodeid==-1){
		return;
	}
	NODE* node=getNode(nodeid);
	if (node->type == TYPE_LEAF) {
		// Print indices of points.
		for (ptrID = node->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId) {
			POINT_BUFFER *ptr= getBUFFER(ptrID);
			for (i = 0; i < ptr->pointCount; i++) {
				fprintf(fp, "%d\n", getPoint((ptr->pointArray)+i)->index);
			}
		}
	} else {
		for (i = 0; i < 4; i++) {
			printLeafPointsDataFile(fp, node->child[i]);
		}
	}
}

// Print x, y coordinates of points in leaf node to file.
void printQuadTreeDataFile(NODEID root, char *outputFile, POINTID pointArray,
		int nPoints) {
	FILE *fp;
	int i;
	fp = fopen(outputFile, "w");
	if (fp == NULL) {
		puts("Cannot open file");
		exit(1);
	}
	fprintf(fp, "%d %d\n", pointRangeX, pointRangeY);
	fprintf(fp, "%d\n", nPoints);
	for (i = 0; i < nPoints; i++) {
		fprintf(fp, "%d %d\n", getPoint(pointArray+i)->x, getPoint(pointArray+i)->y);
	}
	int countNodes = countNodesQuadTree(root);
	fprintf(fp, "%d\n", countNodes);
	int offset = 0;
	int index = 0;
	// Calculate: offset and index per node
	walkNodeTable(NULL, root, &offset, &index);
	// Print
	printTableNodeDataFile(fp, root);
	// Print all points.
	printLeafPointsDataFile(fp, root);
	fclose(fp);
}

// Print point details along with leaf node details.
void printQuadTreeLevel(NODEID nodeid, int level) {
	if(nodeid==-1)
		return;
	NODE* node=getNode(nodeid);
	BUFFID ptrID;
	if (node->level == level) {
		printf(" Node <%d> L=%d [%d %d] (%d %d)\n", node->type, node->level,
				node->posX, node->posY, node->width, node->height);
		if (node->pBuffer != -1) {
			int link = 0;
			int i;
			for (ptrID = node->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId) {
				POINT_BUFFER *ptr= getBUFFER(ptrID);
				printf("  Link %d : ", link);
				for (i = 0; i < ptr->pointCount; i++) {
					printf("(%d : %d,%d) ", getPoint((ptr->pointArray)+i)->index,
							getPoint((ptr->pointArray)+i)->x, getPoint((ptr->pointArray)+i)->y);
				}
				printf("\n");
				link++;
			}
		}

	}
	int i;
	for (i = 0; i < 4; i++) {
		printQuadTreeLevel(node->child[i], level);
	}
}

// Print details of a node passed to the function.
void printQuadTree(NODEID nodeid) {
	if(nodeid==-1)
		return;
	NODE* node=getNode(nodeid);
	printf(" Node <%d> L=%d [%d %d] (%d %d)\n", node->type, node->level,
			node->posX, node->posY, node->width, node->height);
	if (node->pBuffer != -1) {
		BUFFID ptrID;
		int link = 0;
		int i;
		for (ptrID = node->pBuffer; ptrID != -1; ptrID =getBUFFER(ptrID)->nextId) {
			printf("  Link %d : ", link);
			POINT_BUFFER *ptr = getBUFFER(ptrID);
			for (i = 0; i < ptr->pointCount; i++) {
				printf("(%d : %d,%d) ", getPoint((ptr->pointArray)+i)->index,
						getPoint((ptr->pointArray)+i)->x, getPoint((ptr->pointArray)+i)->y);
			}
			printf("\n");
			link++;
		}
	}
	int i;
	for (i = 0; i < 4; i++) {
		printQuadTree(node->child[i]);
	}
}

// Print node details along with point density.
int printNodeStats(NODEID nodeid, int printDetails) {
	int children[4];
	BUFFID ptrID;
	if(nodeid==-1)
		return 0;
	NODE*node=getNode(nodeid);
	int count = 0;
	int sum = 0;
	if (node->pBuffer != -1) {
		int link = 0;
		int i;
		for (ptrID = node->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId) {
			POINT_BUFFER *ptr= getBUFFER(ptrID);
			for (i = 0; i < ptr->pointCount; i++) {
				count++;
			}
		}
		sum++;
	}
	int i;
	if (printDetails) {
		for (i = 0; i < 4; i++) {
			children[i] = 0;
			if (node->child[i])
				children[i] = 1;
		}
		printf(" QuadNode <%d> L=%d [%d %d] (%d %d) %d %f : %d %d %d %d \n",
				node->type, node->level, node->posX, node->posY, node->width,
				node->height, count,
				(float) count / (float) (node->height * node->width),
				children[0], children[1], children[2], children[3]);
	}
	for (i = 0; i < 4; i++) {
		sum += printNodeStats(node->child[i], printDetails);
	}
	return sum;
}

void printQuadTreeStats(NODEID rootNode, unsigned int level, int printDetails) // level =4, printDetails =0
{
	int sum;
	sum = printNodeStats(rootNode, printDetails);
}

/**<************************ Search functions **********************/

/**<************************ Search point ****************************/
/**
 * Function to search point sequentially.
 * 1- Measures time taken to search a point from a point array.
 ** *******************************************************************/
// Function to search a given point from a set of random points.
double searchPoints(POINTID pointArray, int numPoints, POINTID searchArray,
		int numSearches) {
	int s, p;
	int matches = 0;
	long long cost = 0;
	POINT *point, *search;
	// Time the search.
	start_perf_measurement(&timeRecord);
	for (p = 0; p < numPoints; p++) {
		point = getPoint(pointArray+p);
		for (s = 0; s < numSearches; s++) {
			search = getPoint(searchArray+s);
			// Update cost of search and keep count on number of matches.
			cost++;
			if ((search->x == point->x) && (search->y == point->y)) {
				matches++;
			}
		}
	}
	stop_perf_measurement(&timeRecord);
	print_perf_measurement(&timeRecord);
	return (timeRecord.sample);
}

double searchSmartPoints(POINTID pointArray, int numPoints, POINTID searchArray,
		int numSearches) {
	int s, p;
	int matches = 0;
	long long cost = 0;
	int i;
	POINT *point, *search;
	char *maskArray = (char *) malloc(sizeof(char) * numSearches);
	for (i = 0; i < numSearches; i++) {
		maskArray[i] = TRUE;
	}
	// Time the search
	start_perf_measurement(&timeRecord);
	for (p = 0; p < numPoints; p++) {
		point = getPoint(pointArray+p);
		for (s = 0; s < numSearches; s++) {
			if (maskArray[s] == FALSE)
				continue;
			search = getPoint(searchArray+s);
			cost++;
			if ((search->x == point->x) && (search->y == point->y)) {
				matches++;
				maskArray[s] = FALSE;
			}
		}
	}
	stop_perf_measurement(&timeRecord);
	print_perf_measurement(&timeRecord);
	return (timeRecord.sample);
}

/**<********************* Search point in quadtree **********************/
/**
 * Functions to search point using quadtree.
 * 1- Measures time taken to search point in a quadtree.
 * 2- Keeps track of the cost of search and the number of matches.
 * 3- Search function is implemented for both full and adaptive quadtree.
 ** *********************************************************************/
// Find node in which point lies for a full quadtree.
NODEID findQuadTreeNode(NODEID nParentid, struct POINT *p) {
	int posX, posY;
	int x, y;
	int index;
	if(nParentid==-1)
		return nParentid;
	NODE* nParent=getNode(nParentid);
	if (nParent->type == TYPE_LEAF)
		return nParentid;
	// Get the point.
	x = p->x;
	y = p->y;
	// Child width and height
	int width = nParent->width / 2;
	int height = nParent->height / 2;
	for (index = 0; index < 4; index++) {
		switch (index) {
		case 0: // NW
			posX = nParent->posX;
			posY = nParent->posY + height;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return findQuadTreeNode(nParent->child[0], p);
			}
			break;
		case 1: // NE
			posX = nParent->posX + width;
			posY = nParent->posY + height;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return findQuadTreeNode(nParent->child[1], p);
			}
			break;
		case 2: // SW
			posX = nParent->posX;
			posY = nParent->posY;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return findQuadTreeNode(nParent->child[2], p);
			}
			break;
		case 3: // SE
			posX = nParent->posX + width;
			posY = nParent->posY;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return findQuadTreeNode(nParent->child[3], p);
			}
			break;
		}
	}
	return -1;
}

// Function to find node in adaptive quadtree.
NODEID descendQuadTreeNode(NODEID nParentid, struct POINT *p) {
	int posX, posY;
	int x, y;
	int index;
	if (nParentid == -1)
		return -1;
	NODE*nParent=getNode(nParentid);
	// This node has points.
	if (nParent->pBuffer != -1)
		return nParentid;
	// Get the point.
	x = p->x;
	y = p->y;
	// Child width and height
	int width = nParent->width / 2;
	int height = nParent->height / 2;
	for (index = 0; index < 4; index++) {
		switch (index) {
		case 0: // NW
			posX = nParent->posX;
			posY = nParent->posY + height;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return descendQuadTreeNode(nParent->child[0], p);
			}
			break;
		case 1: // NE
			posX = nParent->posX + width;
			posY = nParent->posY + height;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return descendQuadTreeNode(nParent->child[1], p);
			}
			break;
		case 2: // SW
			posX = nParent->posX;
			posY = nParent->posY;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return descendQuadTreeNode(nParent->child[2], p);
			}
			break;
		case 3: // SE
			posX = nParent->posX + width;
			posY = nParent->posY;
			if ((x >= posX) && (x < posX + width) && (y >= posY)
					&& (y < posY + height)) {
				return descendQuadTreeNode(nParent->child[3], p);
			}
			break;
		}
	}
	return -1;
}

// Search full quadtree.
double searchPointsFullQuadTree(NODEID nodeid, POINTID pointArray, int numPoints,
		POINTID searchArray, int numSearches) {
	int i;
	int s;
	int matches = 0;
	int index;
	long long cost = 0;
	POINT point, *search;
	BUFFID ptrID;
	NODEID leaf;
	// Time the search.
	start_perf_measurement(&timeRecord);
	for (s = 0; s < numSearches; s++) {
		search = getPoint(searchArray+s);
		// search the node
		leaf = findQuadTreeNode(nodeid, search);
		if (leaf == -1)
			continue;
		// Get the points from node and check for match.
		for (ptrID = getNode(leaf)->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId) {
			POINT_BUFFER *ptr= getBUFFER(ptrID);
			for (i = 0; i < ptr->pointCount; i++) {
				point = *(getPoint((ptr->pointArray)+i));
				cost++;
				if ((search->x == point.x) && (search->y == point.y)) {
					matches++;
					break;
				}
			}
		}
	}
	stop_perf_measurement(&timeRecord);
	print_perf_measurement(&timeRecord);
	return (timeRecord.sample);
}

// Search adaptive quadtree.
double searchPointsAdaptiveQuadTree(NODEID root, POINTID pointArray,
		int numPoints, POINTID searchArray, int numSearches) {
	int i;
	int s;
	int matches = 0;
	int index;
	long long cost = 0;
	POINT point,* search;
	BUFFID ptrID;
	NODEID node;
	// Time the search.
	start_perf_measurement(&timeRecord);
	for (s = 0; s < numSearches; s++) {
		search = getPoint(searchArray+s);
		// Search the node.
		node = descendQuadTreeNode(root, search);
		// No information
		if (node == -1) {
			continue;
		}
		// Get the points from node and check for match.
		for (ptrID = getNode(node)->pBuffer; ptrID != -1; ptrID = getBUFFER(ptrID)->nextId) {
			POINT_BUFFER *ptr= getBUFFER(ptrID);
			for (i = 0; i < ptr->pointCount; i++) {
				point = *(getPoint((ptr->pointArray)+i));
				cost++;
				if ((search->x == point.x) && (search->y == point.y)) {
					matches++;
					continue;
				}
			}
		}
	}
	stop_perf_measurement(&timeRecord);
	print_perf_measurement(&timeRecord);
	return (timeRecord.sample);
}

/**<************************ Support system ***************************/
//  Get data points from file.
POINTID createFilePoints(char *inputFileName) {
	FILE *fdin;
	if (inputFileName == NULL) {
		printf("Error opening file\n");
		exit(1);
	}
	fdin = fopen(inputFileName, "r");
	if (fdin == NULL) {
		printf("Error opening file\n");
		exit(1);
	}
	if (fscanf(fdin, "%d %d\n", &pointRangeX, &pointRangeY) != 2) {
		printf("Error point file\n");
		exit(1);
	}
	if (fscanf(fdin, "%d\n", &numPoints) != 1) {
		printf("Error point file\n");
		exit(1);
	}
	POINTID pointArray = getPointArray(numPoints);
	int index;
	int x, y;
	for (index = 0; index < numPoints; index++) {
		if (fscanf(fdin, "%d %d\n", &x, &y) != 2) {
			printf("Error point file\n");
			exit(1);
		}
		POINT*p=getPoint(pointArray+index);
		p->x = x;
		p->y = y;
		p->index = index;
	}
	return pointArray;
}

// Create points randomly or read from file.
POINTID createPoints(int numPoints) {
	POINTID pointArray;
	if (pointMode == MODE_RANDOM) {
		pointArray = createRandomPoints(numPoints, rangeSize);
	}
	else {
		pointArray = createFilePoints(inputPointFileName);
	}
	return pointArray;
}

static void usage(char *argv0) {
	char *help = "Usage: %s [switches] -n num_points\n"
			"       -s number_search_points\n"
			"       -l number_levels\n"
			"       -b bucket_size\n"
			"       -p             : print quadtree \n"
			"       -h             : print this help information\n";
	fprintf(stderr, help, argv0);
	exit(-1);
}

// compilation options.
void setup(int argc, char **argv) {
	int opt;
	while ((opt = getopt(argc, argv, "n:l:s:b:r:i:o:aph")) != EOF) {
		switch (opt) {
		case 'n':
			numPoints = atoi(optarg);
			break;
		case 'l':
			numLevels = atoi(optarg);
			break;
		case 's':
			numSearches = atoi(optarg);
			break;
		case 'b':
			bucketSize = atoi(optarg);
			break;
		case 'r':
			rangeSize = atoi(optarg);
			break;
		case 'i':
			inputPointFileName = optarg;
			pointMode = MODE_FILE;
			break;
		case 'o':
			outputTreeFileName = optarg;
			outputTree = 1;
			break;
		case 'p':
			printTree = 1;
			break;
		case 'a':
			quadTreeMode = BUILD_ADAPTIVE;
			break;
		case 'h':
		default:
			usage(argv[0]);
			break;
		}
	}
}


/**<****************** Kernel function ************************/
/**
 * Iterative BFS traversal of quadtree to find points inside polygon.
 * 1- Start traversal at level 3.
 * 2- Assign warp to polygon and thread within each warp to quadtree nodes.
 * 3- Check if node overlaps polygon.
 * 4- If node overlaps, then traverse tree from this node.
 * 5- If the bottom of the tree is reached, then classify nodes based on kind of overlap.
 * 6- Completely overlapping nodes - Get the range of points from node boundary.
 * 7- Partially overlapping nodes - Check every point in node with polygon boundary.
 * 8- Get the range and store.
 * 9- Loop over to evaluate next set of polygons.
 ** **************************************************************/
// Function to check if a polygon is smaller than a given node.
__device__ bool searchPolygonCUDA(int x0, int x1, int xp0, int xp1, int y0, int y1, int yp0, int yp1)
{
	bool N0 = false;
	bool N1 = false;
	bool N2 = false;
	bool N3 = false;
	// Check if a polygon is within a node
	if ((xp0 >= x0) && ((xp0) <= (x1)) && (yp0 >= y0) && ((yp0) <= (y1))) {
		N0 = true;
	}
	if ((xp1 >= x0) && ((xp1) <= (x1)) && (yp0 >= y0) && ((yp0) <= (y1))) {
		N1 = true;
	}
	if ((xp0 >= x0) && ((xp0) <= (x1)) && (yp1 >= y0) && ((yp1) <= (y1))) {
		N2 = true;
	}
	if ((xp1 >= x0) && ((xp1) <= (x1)) && (yp1 >= y0) && ((yp1) <= (y1))) {
		N3 = true;
	}
	if((N0 == false) && (N1 == false) && (N2 == false) && (N3 == false))
	{
		return true;
	}
	return false;
}

// Function to check completely overlapping conditions.
__device__ void NodeCheckComplete(NODEID nodeid, NODE node, Polygon indexOfPolyArray, int**complete, int**partial, int**notN) {
	if(nodeid != NODEID(-1)){
		// x, y coordinates, width and height of node.
		int x0, y0, x1, y1, xp0, yp0, xp1, yp1, w, h;
		int i, j, numOfPoints;
		bool P0 = false;
		bool P1 = false;
		bool P2 = false;
		bool P3 = false;
		// x, y coordinates, width and height of polygon.
		xp0 = indexOfPolyArray.xmin;
		yp0 = indexOfPolyArray.ymin;
		xp1 = indexOfPolyArray.xmax;
		yp1 = indexOfPolyArray.ymax;
		x0 = node.posX;
		y0 = node.posY;
		x1 = x0 + node.width;
		y1 = y0 + node.height;
		// Check for completely overlapping node.
		if ((x0 >= xp0) && ((x0) <= (xp1)) && (y0 >= yp0) && ((y0) <= (yp1))) {
			P0 = true;
		}
		if ((x1 >= xp0) && ((x1) <= (xp1)) && (y0 >= yp0) && ((y0) <= (yp1))) {
			P1 = true;
		}
		if ((x0 >= xp0) && ((x0) <= (xp1)) && (y1 >= yp0) && ((y1) <= (yp1))) {
			P2 = true;
		}
		if ((x1 >= xp0) && ((x1) <= (xp1)) && (y1 >= yp0) && ((y1) <= (yp1))) {
			P3 = true;
		}
		// If all corners of a node is within a polygon, then classify node as completely overlapping.
		if ((P0 == true) && (P1 == true) && (P2 == true)
				&& (P3 == true)) {
			**complete = 1;
		}
		// If all corners of a node is not within a polygon, then classify node as not overlapping.
		else if ((P0 == false) && (P1 == false) && (P2 == false)
				&& (P3 == false) &&  (searchPolygonCUDA(x0, x1, xp0, xp1, y0, y1, yp0, yp1) ==  true))
		{
			**notN = 1;

		}
		// Classify remaining nodes as partially overlapping.
		else{
			**partial = 1;
		}
	}

}

// Function to check not overlapping conditions.
__device__ void NodeCheckNotOverlap(NODEID nodeid, NODE node, Polygon indexOfPolyArray, int**complete, int**partial, int**notN) {
	if(nodeid != (-1)){
		int x0, y0, x1, y1, xp0, yp0, xp1, yp1, w, h;
		int i, j, numOfPoints;
		bool P0 = false;
		bool P1 = false;
		bool P2 = false;
		bool P3 = false;
		xp0 = indexOfPolyArray.xmin;
		yp0 = indexOfPolyArray.ymin;
		xp1 = indexOfPolyArray.xmax;
		yp1 = indexOfPolyArray.ymax;
		x0 = node.posX;
		y0 = node.posY;
		x1 = x0 + node.width;
		y1 = y0 + node.height;
		// Check for not overlapping node.
		if ((x0 >= xp0) && ((x0) <= (xp1)) && (y0 >= yp0) && ((y0) <= (yp1))) {
			P0 = true;
		}
		if ((x1 >= xp0) && ((x1) <= (xp1)) && (y0 >= yp0) && ((y0) <= (yp1))) {
			P1 = true;
		}
		if ((x0 >= xp0) && ((x0) <= (xp1)) && (y1 >= yp0) && ((y1) <= (yp1))) {
			P2 = true;
		}
		if ((x1 >= xp0) && ((x1) <= (xp1)) && (y1 >= yp0) && ((y1) <= (yp1))) {
			P3 = true;
		}
		if ((P0 == false) && (P1 == false) && (P2 == false)
				&& (P3 == false) &&  (searchPolygonCUDA(x0, x1, xp0, xp1, y0, y1, yp0, yp1) ==  true))
		{
			**notN = 1;
		}
		else
		{
			**notN = 0;
		}
	}
}

// Iterative BFS traversal of quadtree to find points inside polygon.
__global__ void searchOverlapNodeCUDA(NODE* d_NODE, NODEID* d_NODE_In, Polygon* d_Polygon, int d_nodeCount, int d_level3Count, int d_numPolygon, int QuadtreeLevel, POINT* d_POINT, POINT_BUFFER* d_POINT_BUFFER, CP* d_cp, PP* d_pp){
	// Global thread index.
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int b_tid = blockDim.x * blockIdx.x + threadIdx.x;
	int val = -1;
	int val1 = -1;
	int *completeNode ,*partialNode, *notNode;
	completeNode = &val;
	partialNode = &val1;
	notNode = &val;
	int NumWarp = 32;
	int threadnumber = (64/32);
	// Keeps count of nodes in the shared memory array.
	__shared__ int nv[32];
	__shared__ int nc[32];
	// Variables to store range of points.
	int minCheckX =  1024;
	int maxCheckX = 0;
	int minCheckY =  1024;
	int maxCheckY = 0;
	int nodeIndexPP = 0;
	int numlevelsQ = 1;
	// Initialize array.
	if(threadIdx.x < NumWarp)
	{
		nv[threadIdx.x] = 0;
		nc[threadIdx.x] = 0;
	}
	// one shared memory array per polygon with array size of 64.
	__shared__ int boundary[50*64];
	// Query polygon global Id.
	int qid1 = tid/32;
	// Query polygon local Id.
	int qidd = threadIdx.x / 32;
	// Starting local index of query Polygon.
	int qb = (threadIdx.x / 32)*64;
	// Thread index within a warp.
	int lane_id = tid%32;
	int nid;
	int iter = 0;
	int x0, y0, x1, y1, xp0, yp0, xp1, yp1;
	// Initialize shared memory array
	if(threadIdx.x < (64*50))
	{
		boundary[threadIdx.x] = -1;
	}
	// Assign warp to a polygon and loop till all polygons are evaluated.
	for(int qid = qid1; qid < d_numPolygon; qid+=(32*gridDim.x))
	{
		// Boundary check for warp Id.
		if(qid < d_numPolygon)
		{
			// Initialize arrays to store completely and partially overlapping node points.
			d_cp[qid].CPcount = 0;
			d_pp[qid].PPcount = 0;
			// Boundary of polygon
			xp0 = d_Polygon[qid].xmin;
			yp0 = d_Polygon[qid].ymin;
			xp1 = d_Polygon[qid].xmax;
			yp1 = d_Polygon[qid].ymax;
			// Start traversal at level 3 of quadtree and check for overlap conditions.
			if(lane_id < d_level3Count){
				int nodeIndex = d_NODE_In[lane_id];
				if(nodeIndex != -1)
				{
					NodeCheckNotOverlap(d_NODE_In[lane_id], d_NODE[nodeIndex], d_Polygon[qid], &completeNode, &partialNode, &notNode);
					if((*notNode) == 0)
					{
						// Add nodes that satisfy query to shared memory array.
						boundary[(qb + atomicAdd(&nc[qidd], 1))] = d_NODE_In[lane_id];
						notNode = &val;
					}
				}
			}
			while(iter < numlevelsQ){
				__syncthreads();
				// Assign node count to local variable.
				int nb = nc[qidd];
				nc[qidd] = 0;
				// Boundary check to launch threads based on the nodes in the array.
				if (lane_id < nb){
					nid = boundary[qb + lane_id];
					// Initialize array after each iteration.
					if(threadIdx.x < (64*50))
					{
						boundary[threadIdx.x] = -1;
					}
					for(int i = 0; i < 4; i++)
					{
						if(nid != -1)
						{
							// Check for overlap conditions for child nodes.
							NodeCheckNotOverlap(d_NODE[nid].child[i], d_NODE[d_NODE[nid].child[i]], d_Polygon[qid], &completeNode, &partialNode, &notNode);
							if((*notNode) == 0) {
								if(nid != -1){

									// Add nodes that satisfy the query to array.
									boundary[(qb + atomicAdd(&nc[qidd],1))] = d_NODE[nid].child[i];
									notNode = &val;

								}
							}
						}
					}
				}
				iter = iter + 1;
			}
			__syncthreads();
			// Assign threads in a warp to nodes and get the points within a node that overlaps a polygon.
			for(int id = lane_id; id < nc[qidd]; id = id + 32)
			{
				nid = boundary[qb + id];
				if(nid != -1){
					// Completely overlapping nodes
					NodeCheckComplete(nid, d_NODE[nid], d_Polygon[qid], &completeNode, &partialNode, &notNode);
					if((*completeNode) == 1)
					{

						// Get node boundary and index of the polygon.
						completeNode = &val;
						BUFFID ptrID;
						NODE node = d_NODE[nid];
						if(node.pBuffer)
						{
							d_cp[qid].nodeNumCP[atomicAdd(&(d_cp[qid].CPcount), 1)] = nid;
							d_cp[qid].polygonIndexCP = qid;
						}

					}
					// partially overlapping nodes
					else if (*partialNode == 1)
					{

						partialNode = &val;
						BUFFID ptrID;
						NODE node = d_NODE[nid];
						if(node.pBuffer)
						{
							for (ptrID = node.pBuffer; ptrID != -1; ptrID = d_POINT_BUFFER[ptrID].nextId)
							{
								if(nid != -1)
								{
									// Check every point within the leaf node against polygon boundary.
									POINT_BUFFER ptr = d_POINT_BUFFER[ptrID];
									for (int c = 0; c < (ptr.pointCount); c++)
									{
										if((d_POINT[(ptr.pointArray)+c].x >= xp0) && (d_POINT[(ptr.pointArray)+c].x <= xp1) && (d_POINT[(ptr.pointArray)+c].y >= yp0) && (d_POINT[(ptr.pointArray)+c].y <= yp1))
										{
											// Store the range of points.
											if(minCheckX > d_POINT[(ptr.pointArray)+c].x )
											{
												minCheckX = d_POINT[(ptr.pointArray)+c].x;
											}
											if(maxCheckX < d_POINT[(ptr.pointArray)+c].x )
											{
												maxCheckX = d_POINT[(ptr.pointArray)+c].x;
											}
											if(minCheckY > d_POINT[(ptr.pointArray)+c].y )
											{
												minCheckY = d_POINT[(ptr.pointArray)+c].y;
											}
											if(maxCheckY < d_POINT[(ptr.pointArray)+c].y )
											{
												maxCheckY = d_POINT[(ptr.pointArray)+c].y;
											}
										}
									}
								}

								nodeIndexPP = atomicAdd(&(d_pp[qid].PPcount), 1);
								d_pp[qid].nodePP_array[nodeIndexPP].xminPP = minCheckX;
								d_pp[qid].nodePP_array[nodeIndexPP].yminPP = maxCheckX;
								d_pp[qid].nodePP_array[nodeIndexPP].xmaxPP = minCheckY;
								d_pp[qid].nodePP_array[nodeIndexPP].ymaxPP = maxCheckY;
								d_pp[qid].polygonIndexPP = qid;
							}
						}

					}
				}
			}
		}

	}

}

/**<************************ Main function ***************************/
/**
 * Two techniques to build QuadTrees
 * 1- full : extend all the way down, only leafs hold points
 *         : counts are kept at intermediate levels
 *         : nulls are still used to know where points are.
 * 2- adaptive : items are pushed around as needed to form tree
 *         : points of LIMIT pushed down.
 ** ******************************************************************/
int main(int argc, char **argv) {
	setup(argc, argv);
	// Host variables.
	int index;
	NODEID rootNode;
	struct Polygon *randomPoly;
	// Device variable declaration.
	Polygon *d_Polygon;
	CP *d_cp;
	PP *d_pp;
	PP *d_hostPoint;
	// Preallocate memory for all objects in CPU.
	allocatePointMemory();
	allocateNodeMemory();
	allocatePOINTBUFFERMemory();
	d_hostPoint=(PP*)malloc(numPolygon*sizeof(PP));
	// Create random points and polygon.
	POINTID pointArray = createPoints(numPoints);
	randomPoly = createRandomPolygon(numPolygon, rangeSize);
	// Get memory for root node.
	rootNode = getNewNode();
	// Start node : root
	setNode(rootNode, 0, 0, rangeSize, rangeSize, TYPE_ROOT, 0);
	// Create the quadtree.
	buildQuadTree(rootNode, numLevels, pointArray, numPoints, quadTreeMode);
	//Time quadtree construction.
	double buildTime = timeRecord.sample;
	printf("QuadTreeBuild Time : %f\n", buildTime);
	// Print the quadtree.
	if (printTree) {
		printQuadTree(rootNode);
	}
	printQuadTreeStats(rootNode, numLevels, 0);
	if (outputTree)
	{
		printQuadTreeDataFile(rootNode, outputTreeFileName, pointArray, numPoints);
	}
	// Search section
	if (numSearches > 0)
	{
		// Create some search points.
		POINTID searchArray = createRandomPoints(numSearches, rangeSize);
		// Search points in an array.
		double baseTime = searchPoints(pointArray, numPoints, searchArray,
				numSearches);
		double smartTime = searchSmartPoints(pointArray, numPoints, searchArray,
				numSearches);
		// Search points using quadtree.
		double quadTime;
		if (quadTreeMode == BUILD_FULL)
			quadTime = searchPointsFullQuadTree(rootNode, pointArray, numPoints,
					searchArray, numSearches);
		else
		{
			quadTime = searchPointsAdaptiveQuadTree(rootNode, pointArray,
					numPoints, searchArray, numSearches);
		}
	}
	//CUDA memory allocation.
	allocatePointMemoryCuda();
	allocatePoint_BufferMemoryCuda();
	allocateNodeMemoryCuda();
	cudaMalloc((void**)&d_Polygon, sizeof(Polygon)*numPolygon);
	cudaMalloc((void**)&d_cp, sizeof(CP)*numPolygon);
	cudaError_t err = cudaMalloc((void**)&d_pp, sizeof(PP)*numPolygon);
	// Time the kernel execution.
	float diff_GPU;
	cudaEvent_t start_GPU, stop_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&stop_GPU);
	cudaEventRecord(start_GPU, 0);
	// Copy nodes at level 3.
	levelThreeNode();
	allocateNodeIDMemoryCuda();
	int d_nodeCount = nodeArr.currCount;
	int d_level3Count = Level3NodeArray.count;
	//Polygon CUDA memory allocation.
	PointCudaCopy();
	Point_BufferCudaCopy();
	NodeCudaCopy();
	NodeIDCudaCopy();
	cudaMemcpy(d_Polygon, randomPoly, sizeof(Polygon)*numPolygon, cudaMemcpyHostToDevice);
	//Launch kernel with 65535 blocks and 1024 threads per block.
	searchOverlapNodeCUDA<<<65535, 1024>>>(d_NODE, d_NODE_In, d_Polygon, d_nodeCount, d_level3Count, numPolygon, numLevels, d_POINT, d_POINT_BUFFER, d_cp, d_pp);
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != CUDA_SUCCESS){
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
	}
	cudaEventRecord(stop_GPU, 0);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&diff_GPU, start_GPU, stop_GPU);
	// Memory transfer from device to host.
	cudaMemcpy(d_hostPoint, d_pp, sizeof(PP)*numPolygon, cudaMemcpyDeviceToHost);
	// Calculation of number polygons per second.
	int NumberofPolygons = numPolygon / (diff_GPU/1000);
	printf("%d\n", NumberofPolygons);
	// Destroy CUDA event.
	cudaEventDestroy(start_GPU);
	cudaEventDestroy(stop_GPU);
	// Free CUDA memory.
	cudaFree(d_Polygon);
	cudaFree(d_POINT_BUFFER);
	cudaFree(d_POINT);
	cudaFree(d_NODE);
	return 0;
}

