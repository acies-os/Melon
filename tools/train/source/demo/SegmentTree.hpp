//
// Created by 王启鹏 on 2022/7/2.
//

#ifndef CTEST_SEGMENTTREE_HPP
#define CTEST_SEGMENTTREE_HPP


#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

using namespace std;
int N, Q;

template<class T, class F>
class SegmentTree {
public:
    SegmentTree(int left, int right);
    void insert(int pos, T k, int index=0);
    void insert(pair<int, int> p, T k, int index=0, bool updateParent=true);
    T query(int left, int right, int index=0);
    void show();
private:
    struct Node;
    vector<shared_ptr<Node>> tree;
    int initialize(int left, int right, int index=0);
    void updateNode(T k, int index);
};



#endif //CTEST_SEGMENTTREE_HPP
