#!/usr/bin/env python3
"""
Defines build decision tree training
Classes:
    Node: no leaf in tree
    Leaf: leaf node and inherit from Node
    Decision_Tree: main clas
"""
import numpy as np


class Node:
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

    def count_nodes(self, only_leaves=False):
        """
        Calculate numbers of nodes
        Args:
            only_leaves: if true, count only leaf
        Returns:
            count of nodes
        """
        if self.is_leaf:
            return 1 if only_leaves else 0

        left_count = (self.left_child.count_nodes(only_leaves)
                      if self.left_child else 0)
        right_count = (self.right_child.count_nodes(only_leaves)
                       if self.right_child else 0)

        return left_count + right_count + (0 if only_leaves else 1)


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth


class Leaf(Node):
    """
    A leaf node in a decision tree inheriting from class Node
    Attributes:
        value: the value for leaf node
        depth: the depth of the leaf node
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def pred(self, x):
        """
        Predict the value for a single input using the node's children.
        """
        return self.value

    def update_indicator(self):
        """
        Placeholder for updating any indicator related to the leaf.
        """
        pass


class Decision_Tree():
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

    def np_extrema(self, arr):
        """
        Get the minumin and maximum in array by numpy
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        choose the feature from data a random
        and choose threshold for split by it
        input:
            node: has information at sub_population
        Return:
            feature, threshold when diff!=0
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """
        Training model decision tree
        Args:
            explanatory: database input
            target: the output of database (label)
            verbose: by default is 0 and is optional
        Return:
            if verbose = 1 return number of node, leaf, depth
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_creterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype="bool")
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory,self.target)}""")

    def fit_node(self, node):
        """
        -This function is part of build decision tree
        -build tree step by step with split by
         two part left_population and right_population
        """
        node.feature, node.threshold = self.split_criterion(node)
        left_population = node.sub_population & (
            (self.explanatory[:, node.feature] > node.threshold))

        right_population = node.sub_population & ~left_population

        if len(left_population) != len(self.target):
            left_population = np.pad(left_population,
                                     (0, len(self.target) - len(
                                         self.left_population)),
                                     'constant', constant_values=(0)
                                     )
            right_population = np.pad(right_population,
                                      (0, len(self.target) - len(
                                          self.right_population)),
                                      'constant', constant_values=(0)
                                      )

        # Is left node a leaf ?
        is_left_leaf = (
            node.depth == self.max_depth - 1 or
            np.sum(left_population) <= self.min_pop or
            np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?

        is_right_leaf = (
            node.depth == self.max_depth - 1 or
            np.sum(right_population) <= self.min_pop or
            np.unique(self.target[right_population]).size == 1
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        create leaf when dont have child
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        create node when have child
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def update_predict(self):
        """omputes the prediction"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def update_bounds(self):
        """Update the bounds (lower and upper)"""
        pass

    def get_leaves(self):
        """
        Returns all leaf nodes in the decision tree.
        """
        leaves = []

        def traverse(node):
            """Travese"""
            if node.is_leaf:
                leaves.append(node)
            else:
                if node.left_child:
                    traverse(node.left_child)
                if node.right_child:
                    traverse(node.right_child)

        traverse(self.root)
        return leaves

    def depth(self, node=None):
        """
        Depth of tree
        """
        if node is None:
            node = self.root

        if node.left_child is None and node.right_child is None:
            return 1

        left_depth = 0
        right_depth = 0

        if node.left_child is not None:
            left_depth = self.depth(node.left_child)

        if node.right_child is not None:
            right_depth = self.depth(node.right_child)

        return max(left_depth, right_depth)

    def count_nodes(self, only_leaves=False):
        """
        calculate nodes
        """
        return self.root.count_nodes(only_leaves)

    def pred(self, x):
        """
        Predict the value for a single input using the node's children.
        """
        node = self.root
        while not node.is_leaf:
            if x[node.feature] > node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        return node.value

    def count_nodes(self, only_leaves=False):
        """
        append function count_node from class Node
        """
        if self.root:
            return self.root.count_nodes(only_leaves)
        return 0

    def accuracy(self, test_explanatory, test_target):
        """
        testing tree
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size
