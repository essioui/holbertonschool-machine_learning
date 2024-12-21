#!/usr/bin/env python3
"""
5-build_decision_tree : the update_bounds method
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
        """
        Collect all leaf nodes below the current node.
        """
        if self.is_leaf:
            return [self]

        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())

        return leaves

    def update_bounds_below(self):
        """
        Update the bounds (lower and upper) for all nodes below this node.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                feature = self.feature
                thresold = self.threshold

                if child == self.left_child:
                    child.lower[feature] = thresold
                else:
                    child.upper[feature] = thresold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        the indicator function for a given node, denoted as “n.
        """
        def is_large_enough(x):
            return np.all(
                np.array(
                    [np.greater(
                        x[:, key], self.lower[key])
                        for key in self.lower.keys()]),
                axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array(
                    [np.less_equal(
                        x[:, key], self.upper[key])
                        for key in self.upper.keys()]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf(Node):
    """
    A leaf node in a decision tree inheriting from class Node
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def update_bounds_below(self):
        """
        Leaves do not have child nodes to propagate bounds.
        """
        pass


class Decision_Tree():
    """
    Decision tree of classification or regression
    """
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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
        Retrieve all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds"""
        self.root.update_bounds_below()
