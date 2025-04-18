#!/usr/bin/env python3
"""
0. Depth of a decision tree
Defines classes and methods in decision tree
Classes:
    Node: no leaf in tree
    Leaf: leaf node and inheritfrom Node
    Decision_Tree: main class
"""
import numpy as np


class Node:
    """
    Structure of decision tree
    Attributes:
        feature: the property use for partition
        threshold: build nodes
        left_child, right_child: branch from nodes
        is_leaf: if is leaf or no
        depth: deeping of tree
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculate  the maximum depth of tree
            is_leaf: the deep is leaf
            left_depth, right_depth: calculate the deep of tree
        Return: the max between left deep and right deep
        """
        if self.is_leaf:
            return self.depth
        else:
            left_depth = (self.left_child.max_depth_below() if self.left_child
                          else self.depth)
            right_depth = (self.right_child.max_depth_below()
                           if self.right_child else self.depth)
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes
        Args:
            only_leaves: if true, counts only leaf node
        Return:
            count of node
        """
        if self.is_leaf:
            return 1 if only_leaves else 0
        else:
            left_count = (self.left_child.count_nodes_below(only_leaves)
                          if self.left_child else 0)
            right_count = (self.right_child.count_nodes_below(only_leaves)
                           if self.right_child else 0)
            return left_count + right_count + (0 if only_leaves else 1)


class Leaf(Node):
    """
    a leaf node in decision tree inhirt from class Node
    attributes:
        value: the value for leaf node
        depth: the depth of the leaf node
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return depth is the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 for leaf node"""
        return 1


class Decision_Tree():
    """
    Decision tree of classification or regression
    Attributes:
        max_depth: the maximum depth of tree
        min_pop: the minimum number for split tree
        seed: used for random number
        split_criterion: used to split a node
        root: the root node of tree
    return the depth
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodein tree
        Args:
            only_leaves: if true, count only the leaf nodes
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)
