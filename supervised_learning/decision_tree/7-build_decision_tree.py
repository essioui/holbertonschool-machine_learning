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

    def pred(self, x):
        """
        Predict the value for a single input using the node's children.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def get_leaves_below(self):
        """Return the leaf"""
        return [self]


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

    def get_leaves(self):
        """
        Print the leaves from root node
        """
        return self.root.get_leaves_below()

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

    def update_bounds(self):
        """Update the bounds"""
        self.root.update_bounds_below()

    def pred(self, x):
        """
        Predict the value for a single input using the node's children.
        """
        return self.root.pred(x)

    def update_predict(self):
        """omputes the prediction"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([self.pred(x) for x in A])

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
            feature_min, feature_max = self.np_extrema(self.explanatory
                                                       [:, feature]
                                                       [node.sub_population])
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
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """
        -This function is part of build decision tree
        -build tree step by step with split by
         two part left_population and right_population
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & \
            (self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population
        if len(left_population) != len(self.target):
            left_population = np.pad(
                left_population,
                (0, len(self.target) - len(self.left_population)),
                'constant', constant_values=(0)
            )
        if len(right_population) != len(self.target):
            right_population = np.pad(
                right_population,
                (0, len(self.target) - len(self.right_population)),
                'constant', constant_values=(0)
            )
        is_left_leaf = (
            node.depth == self.max_depth - 1 or
            np.sum(left_population) <= self.min_pop or
            np.unique(self.target[left_population]).size == 1
        )
        is_right_leaf = (
            node.depth == self.max_depth - 1 or
            np.sum(right_population) <= self.min_pop or
            np.unique(self.target[right_population]).size == 1
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            node.left_child.depth = node.depth + 1
            self.fit_node(node.left_child)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            node.right_child.depth = node.depth + 1
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
        A = self.target[sub_population]
        B, C = np.unique(A, return_counts=True)
        value = B[np.argmax(C)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a non-leaf child node.

        Parameters
        node : Node
            The parent node.
        sub_population : array-like
            The sub-population for the child node.

        Returns
        Node
            The created non-leaf child node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        testing tree
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target))/test_target.size
