//
// Created by 王启鹏 on 2022/7/2.
//

#ifndef CTEST_SEGMENTTREE_HPP
#define CTEST_SEGMENTTREE_HPP


#include <iostream>
#include <vector>
#include <algorithm>
// #include <functional>

#include <memory>
#include <limits>

using namespace std;
// int N, Q;

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

template<class T, class F>
struct SegmentTree<T, F>::Node {
    // (Jinyang)
    // In a Segment Tree, each node has a range [left, right]. The left and
    // right here denote the range and these are not the children. The left
    // child and right child are acccessed by indexing. Suppose the current
    // node's index is i:
    //     left child: 2 * i + 1
    //     right child: 2 * i + 2
    int left;
    int right;
    T targetValue;

    Node() {
        left = right = -1;
        // if F = std::less, less<T>()(0, 1) => 0 < 1 => true => min seg tree
        // if F = std::greater => max seg tree
        if (F()(0, 1)) {
            targetValue = numeric_limits<T>::max();
        } else {
            targetValue = numeric_limits<T>::min();
        }
    }
};

template<class T, class F>
SegmentTree<T, F>::SegmentTree(int left, int right) {
    if (left > right) {
        return;
    }
    // (Jinyang)
    // "<< 2" == "* 4"
    // A property of Segment Tree is that they need 4*n (at most) to work with an
    // array of size n
    tree.resize((right - left + 1) << 2);
    for (int i = 0; i < tree.size(); ++i) {
        tree[i] = make_shared<Node>();
    }

    // (Jinyang)
    // the initialize() function will return the largest index used; +1 will be
    // the size. Note that due to the indexing scheme (left: 2*i+1; right:
    // 2*i+2), there will be unused indicies. This last resize simply trims off
    // unused slots at the *end* of the vec, but does not eliminates the holes
    // in the vec.
    tree.resize(initialize(left, right) + 1);
}

// (Jinyang)
// this only set up the range for each node; it does NOT populate the values
template<class T, class F>
int SegmentTree<T, F>::initialize(int left, int right, int index) {
    // (Jinyang)
    // set the node's range to [left, right], inclusive
    tree[index]->left = left;
    tree[index]->right = right;

    // (Jinyang)
    // leaf node; stop recursion
    if (left == right) {
        return index;
    }

    // (Jinyang)
    // If not leaf node, it's an internal node.
    // The mid point will be (l+r)/2
    // left child:
    //    range: [l, mid]
    //    index: 2*i+1
    // right child:
    //    range: [mid+1, r]
    //    index: 2*i+2
    int mid = (left + right) >> 1;
    int l = initialize(left, mid, (index << 1) + 1);
    int r = initialize(mid + 1, right, (index << 1) + 2);

    // (Jinyang)
    // return the largest index in the tree
    return max(l, r);
}

template<class T, class F>
void SegmentTree<T, F>::insert(int pos, T k, int index) {
    int mid = (tree[index]->left + tree[index]->right) >> 1;
    // (Jinyang)
    // leaf node;
    // assert left == right == pos
    // insert the value k here
    if (tree[index]->left == tree[index]->right) {
        tree[index]->targetValue = k;
        return;
    }
    if (F()(k, tree[index]->targetValue)) {
        tree[index]->targetValue = k;
    }
    if (pos <= mid) {
        insert(pos, k, (index << 1) + 1);
    } else {
        insert(pos, k, (index << 1) + 2);
    }
}


template<class T, class F>
void SegmentTree<T, F>::insert(pair<int, int> p, T k, int index, bool updateParent) {
    printf("begin insert: node=[%d, %d]\tsection=[%d, %d]\n", tree[index]->left, tree[index]->right, p.first, p.second);
    if (p.first > tree[index]->right || p.second <= tree[index]->left) {
        // no intersection, but this block is dead code.
        return;
    }
    if (tree[index]->left == p.first && tree[index]->right == p.second) {
        printf("update [%d, %d]\n", p.first, p.second);
        updateNode(k, index);
        return;
    }
    if (F()(k,  tree[index]->targetValue)) {
        tree[index]->targetValue = k;
    }
    p.first = max(p.first, tree[index]->left);
    p.second = min(p.second, tree[index]->right);
    int mid = (tree[index]->left + tree[index]->right) >> 1;
    printf("after intersection: node=[%d, %d]\tmid=%d\tsection=[%d, %d]\n", tree[index]->left, tree[index]->right, mid, p.first, p.second);
    if (mid < p.first) {
        insert(p, k, (index << 1) + 2, false);
    } else if (mid >= p.second) {
        insert(p, k, (index << 1) + 1, false);
    } else {
        insert(make_pair(p.first, mid), k, (index << 1) + 1, false);
        insert(make_pair(mid + 1, p.second), k, (index << 1) + 2, false);
    }
    while (index >= 0 && updateParent) {
        if (F()(tree[index]->targetValue, tree[index >> 1]->targetValue)) {
            tree[index >> 1]->targetValue = tree[index]->targetValue;
        } else {
            break;
        }
    }
}


template<class T, class F>
void SegmentTree<T, F>::updateNode(T k, int index) {
    tree[index]->targetValue = k;
    if (tree[index]->left == tree[index]->right) {
        return;
    }
    updateNode(k, (index << 1) + 1);
    updateNode(k, (index << 1) + 2);
}

template<class T, class F>
T SegmentTree<T, F>::query(int left, int right, int index) {
    if (tree[index]->left > right || tree[index]->right < left || left > right) {
        if (F()(0, 1)) {
            return numeric_limits<T>::max();
        } else {
            return numeric_limits<T>::min();
        }
    } else if (tree[index]->left >= left && tree[index]->right <= right) {
        return tree[index]->targetValue;
    } else {
        T l = query(left, right, (index << 1) + 1);
        T r = query(left, right, (index << 1) + 2);
        if (F()(l, r)) return l;
        else return r;
    }
}

template<class T, class F>
void SegmentTree<T, F>::show() {
    for (auto node: tree) {
        cout << node->left << " " << node->right << " " << node->targetValue << "\n";
    }
    cout << "\n";
}

#endif //CTEST_SEGMENTTREE_HPP
