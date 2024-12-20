#!/usr/bin/env python3
"""
Defines build_decision_tree
Classes:
    Node: no leaf in tree
    Leaf: leaf node and inherit from Node
    Decision_Tree: main clas
"""
import numpy as np


class Node:
    """
    Structure of decision tree
    Attributes:
        feature: the property used for partition
        threshold: build nodes
        left_child, right_child: branches from nodes
        is_leaf: if the node is a leaf
        depth: depth of tree
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

    def get_leaves_below(self):
        """returns the list of all leaves of the tree"""
        if self.is_leaf:
            return f"-> leaf [value={self.value}]"

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def __str__(self):
        """Return the result in string"""
        if self.is_leaf:
            return f"Leaf(value={self.value})"


class Leaf(Node):
    """
    A leaf node in a decision tree inheriting from class Node
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def get_leaves_below(self):
        """Return the leaf"""
        return [self]


class Decision_Tree():
    """
    Decision tree of classification or regression
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

    def get_leaves(self):
        """
        Print the leaves from root node
        """
        return self.root.get_leaves_below()
