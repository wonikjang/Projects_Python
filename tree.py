#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:19:22 2018

@author: wonikJang
"""

# ======= Concepts 

# 1. (BFS) Breadth-first Search 
# start from the tree root , explores all the neighbor nodes prior to moving on to the nodex at the next depth level

# 2. (DFS) Depth-first Search 
# explore the highest-depth nodes first before forced to backtrack and expand shallower nodes 




# ======= Level Order tree Traversal 

# Recursive Python program for level order traversal of Binary Tree
 
# A node structure
class Node:
 
    # A utility function to create a new node
    def __init__(self, key):
        self.data = key 
        self.left = None
        self.right = None
 
 
# Function to  print level order traversal of tree
def printLevelOrder(root):
    h = height(root)
    print("level is : " +str(h))
    for i in range(1, h+1):
#        print("level is : " +str(i))
        printGivenLevel(root, i)
 
 
# Print nodes at a given level
def printGivenLevel(root , level):
    if root is None:
        return
    if level == 1:
        print("%d" %(root.data) )
    elif level > 1 :
        printGivenLevel(root.right , level-1)
        printGivenLevel(root.left , level-1)

 
 
""" Compute the height of a tree--the number of nodes
    along the longest path from the root node down to
    the farthest leaf node
"""
def height(node):
    if node is None:
        print("node is none!")
        return 0
    else :
        # Compute the height of each subtree 
        print("left cal Starts : ")   
        
        
        
        print(node.left)
        lheight = height(node.left)
        print("lheight : " + str(lheight))
        
        
        
        
        print("right cal Starts : ")
        rheight = height(node.right)
        print("rheight : " + str(rheight))
        print("=============================")
    
        #Use the larger one / return the result (the lower level) + 1 to higher level 
        if lheight >= rheight :
            
            print("============ Left >= Right")
            
            return lheight+1
        else:
            
            print("============ Right > Left")
            
            return rheight+1

# Driver program to test above function
root = Node(1)
root.left = Node(2)
root.right = Node(3)



root.left.left = Node(4)
root.left.right = Node(5)
 
print("Level order traversal of binary tree is -")
printLevelOrder(root)

























# ======= Tree Levels 

# 1.  Declare Tree and node 

class Tree: 
    def __init__(self, root) : 
        self.root = root 
        
#tree = Tree(1)

class Node: 
    def __init__(self, data, children = [] ) :
        self.data = data
        self.children = children 
        

