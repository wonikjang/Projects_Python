# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:45:37 2018

@author: 09732
"""

# ============================== 1. Recurring character return 
 
data = 'ABDCAB' # Input
output = ['A','B'] # Expected Output 

# Looking over the whole string --> O(N^2)
# Using dictionary (hash table) --> O(N)

def getRecur( data ):
    
    dict_count = {}
    
    if len(data) < 1:
        print("empty data")
        return None
    
    # Create dictionary that count occurance
    for char in data:
        if char in dict_count:
            dict_count[char] = dict_count[char] + 1
        else: 
            dict_count[char] = 1 

    # Select recurrent characters only 
    recurrent_list = []
    for k, v in dict_count.items():
        if v >= 2:
            recurrent_list.append(k)

   
    return recurrent_list            
            
output = getRecur(data)
output



# ============================== 2. First non recurring character

def firstNonRecur( data ):
    
    order = []
    dict_count = {}
    
    if len(data) < 1:
        print("empty data")
        return None
    
    # Create dictionary that count occurance
    for char in data:
        if char in dict_count:
            dict_count[char] = dict_count[char] + 1
        else: 
            dict_count[char] = 1 
            order.append(char)
    
    for ele in order:
        if dict_count[ele] == 1:
            return ele
    
output = firstNonRecur(data)
output


# ============================== 3. Array of integer 
# [1, 3, 2, 4] 
# Function that add 1 --> [1 ,3, 2, 5]

# Step 1 : Clarifying question --> Could this array be empty? Can we assume that integer is always between 0 and 9? 
# Step 2 : Explain high level --> Not to code at first  [9,9,9] --> [1,0,0,0]
# Step 3 : List out : Iterative approcah with for loop vs. Recursive approach 
# Step 4 : Ask - Should I start coding?


# Python & Function header 

# [1,3,4] --> [1,3,5] 
# [9,9,9] --> [1,0,0,0]

import numpy as np
#given_array = [1,3,4]
given_array = [1,3,9]
#given_array = [9,9,9]

def add_one(given_array):
    
    carry = 1
#    result = np.zeros( len(given_array) )
    result = [0] * len(given_array) 
    
    for i in range( len(given_array)-1 , -1 , -1 ):
        
        total = given_array[ i ] + carry 
        
        if total == 10:
            carry = 1
        else: 
            carry = 0 
        
        result[ i ] = total % 10 
        
    
    if carry == 1: 
#        result =  np.insert( result, 1) 
        result.insert(0,1)
    
    return result
    
output = add_one(given_array)   
output    
    
    
    
# ============================= 
#class Node:
#    def __init__(self, data=None, left=None, right=None):
#        self.data = data
#        self.left = right
#        self.right = right
        
# Function to insert into Binary Search Tree
#def insert(root, data):
#    if data < root.data:
#        # print "hi"
#        if root.left is None:
#            root.left = Node(data)
#        else:
#            insert(root.left, data)
#    elif data > root.data:
#        if root.right is None:
#            root.right = Node(data)
#        else:
#            insert(root.right, data)
#
#
#root = Node(7)
#insert(root, 5)
#insert(root, 8)
#insert(root, 4)
#insert(root, 6)
#insert(root, 11)

# ============================= Iterative way 

# ================ Data Generation 

class Node:
    def __init__(self, data, children = []):
        self.data = data
        self.children = children


node2_1 = Node(4,None)
node2_2 = Node(5,None)
node2_3 = Node(6,None)
node2_4 = Node(7,None)

node1_1 = Node(2, [node2_1, node2_2] )
node1_2 = Node(3, [node2_3, node2_4] )

node_root = Node(1, [node1_1, node1_2])



# ================= Algorithm : BFS 

queue = []
queue.append( ( 0, node_root )  )

dict_sum = {}

while len(queue) > 0 : 
    
#    (level, node) = queue.get()
    level, node = queue.pop()
    
#    print(level)
#    print(node.data)
    
    if level not in dict_sum:
        dict_sum[level] = node.data
    else: 
        dict_sum[level] += node.data

    
    if node.children:
        for child in node.children:
            queue.append( ( level + 1, child ) )

print("======== ")
print(dict_sum)

# ================= Algortihm : DFS 

#def recursive_len_sum(node):
#    temp = 0
#    if node.symbol:
#        temp += len(node.code)
#    for child in node.children:
#        temp += recursive_len_sum(child)
#    return temp
#
#recursive_len_sum(root)



# ================= Algorithm that can get all possible root to leaf paths.

    
# Recursively find 

def get_all_paths(node, path=None):
    
    paths = []
    if path is None:
        path = []
    path.append(node.data)
    
    if node.children:
        for child in node.children:
            paths.extend(get_all_paths(child, path[:]))
    else:
        paths.append(path)
    return paths

get_all_paths(node_root)

# Iteratively find 

#def iterativeChildren(nodes):
#	results = []
#	while 1:
#		newNodes = []
#		if len(nodes) == 0:
#			break
#		for node in nodes:
#			results.append(node.data)
#			if len(node.children) > 0:
#				for child in node.children:
#					newNodes.append(child)
#		nodes = newNodes
#	return results
#
#iterativeChildren()







