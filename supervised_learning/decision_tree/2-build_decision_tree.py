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

    def left_child_add_prefix(self, text):
        """
        take tree like text and split by "\n"
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        take tree like text and split by "\n"
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def __str__(self):
        """
        Present the nodes not leaf in tree
        Return:
            text: Node [feature=feature, thresold=threshold]
            concatenate between left_text and right_text
        """
        left_text = self.left_child.__str__() if self.left_child else ""
        right_text = self.right_child.__str__() if self.right_child else ""
        text = f"Node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child and self.right_child:
            left_text = self.left_child_add_prefix(left_text)
            right_text = self.right_child_add_prefix(right_text)

        return f"{text}\n{left_text}{right_text}"


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

    def __str__(self):
        """Return the value of leaf ->leaf [value=value]"""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree:
    def __init__(self, root=None):
        """
        Print tree start from root node
        if is_root=False that mean isnt root node
        """
        self.root = root if root else Node(is_root=True)

    def __str__(self):
        """Print tree like text"""
        return self.root.__str__()
