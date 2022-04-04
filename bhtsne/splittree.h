/*
 *  quadtree.h
 *  Header file for a quadtree.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#include <cstdlib>
#include <vector>

#ifndef SPLITTREE_H
#define SPLITREE_H

static inline float min(float x, float y) { return (x <= y ? x : y); }
static inline float max(float x, float y) { return (x <= y ? y : x); }
static inline float abs_d(float x) { return (x <= 0 ? -x : x); }

class Cell {

public:
	float* center;
	float* width;
	int n_dims;
	bool   containsPoint(float point[]);
	~Cell() {
		delete[] center;
		delete[] width;
	}
};


class SplitTree
{

	// Fixed constants
	static const int QT_NODE_CAPACITY = 1;

	// Properties of this node in the tree
	int QT_NO_DIMS;
	bool is_leaf;
	int size;
	int cum_size;

	// Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
	Cell boundary;

	// Indices in this quad tree node, corresponding center-of-mass, and list of all children
	float* data;
	float* center_of_mass;
	int index[QT_NODE_CAPACITY];

	int num_children;
	std::vector<SplitTree*> children;
public:


	SplitTree(float* inp_data, int N, int no_dims);
	SplitTree(SplitTree* inp_parent, float* inp_data, float* mean_Y, float* width_Y);
	~SplitTree();
	void construct(Cell boundary);
	bool insert(int new_index);
	void subdivide();
	void computeNonEdgeForces(int point_index, float theta, float* neg_f, float* sum_Q);
private:

	void init(SplitTree* inp_parent, float* inp_data, float* mean_Y, float* width_Y);
	void fill(int N);
};

#endif
