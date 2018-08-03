# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 09:18:59 2018

@author: 09732
"""

arr = [1,2,4,5,10,20,50]



# ============ 1. Most Frequently Occuring Item  : O(n)


def most_frequent(given_array):
    
    dict_count = {}

    # Initialize variables 
    max_item = None
    max_count = 0
    
    # Exceptional Case 
    if given_array is None:
        return None
    
    # Normal Case 
    else: 
        
        for item in given_array:
            if item not in dict_count.keys():
                dict_count[item] = 1
            else: 
                dict_count[item] += 1
                
            if dict_count[item] > max_count:
                
                # Update max_count 
                max_count = dict_count[item]
                max_item  = item
                
    return max_item 

arr1 = [1,2,4,5,10,20,50 ,1]
most_frequent(arr1)


# =========== 2. Common Element in Two sorted Arrays : O(max(n,m))

a = [1,3,4,6,7,9]
b = [1,2,4,5,9,10]

# Comment: Need sizable data strucuture to save common elements --> List 

def common_elements(a , b):
    
    p1 = 0 
    p2 = 0 
    result = []
    
    while p1 < len(a) and p2 < len(b):
        if a[p1] == b[p2]:
            result.append( a[p1])
            p1 += 1
            p2 += 1
        elif a[p1] > b[p2]:
            p2 += 1
        elif a[p1] < b[p2]:
            p1 += 1
    
    return result

res0 = common_elements(a,b)


# ========== 3. Is one array a rotation of the other ? : O(n) 
# ========== Multiple False Cases are conditioned 

# Assume : No duplicates 

# Comment : first Check the length of a and b 
# Comment : Find where the first item of a is included in b 


a = [1,2,3,4,5,6,7]
b = [4,5,6,7,1,2,3]

#def is_rotation(list1, list2):
#    
#    memo = list1.extend(list1[0])
#    for i in range(len(list1)):
#         list1[i:i+2]
    
def is_rotation(A,B):
    
    # Base Case 
    if len(A) != len(B):
        return False 
    key = A[0]; 
    key_i = -1 # key of b 
    
    for i in range(len(B)):
        if B[i] == key:
            key_i = i 
            break 
    
    if key_i == -1:
        return False
    
    for i in range(len(A)):
        j = (key_i + i) % len(a)
        if a[i] != b[j]:
            return False
    return True
    

is_rotation(a,b)


# ========= 4. Non Repeating chara return : O(n)


def single_char(given_array):
    
    single_item  = None 
    dict_count = {}
    
    if given_array is None:
        return None
    
    for item in given_array:

        if item not in dict_count.keys():
            dict_count[item] = 1
        else: 
            dict_count[item] += 1

            
        if dict_count[item] == 1: 
            single_item = item

        
    return single_item

single_char([1,1,2])


# ========= 5. One Away Strings : O(n)

# Given 2 strings, True / False whether one awaw : 
# 1. Changed (including the same) 2. Eat(One is removed ) 3. Add (One is more )

s1 = 'abcde'
s2 = 'abde'


def one_differ( s1, s2 ):
    
    p1 = 0 
    p2 = 0
    count = 0 
    
    while p1 < len(s1) and p2 < len(s2):
    
        # Regarding Cases 
        
        # Difference in length > 2
        if abs( len(s1) - len(s2) ) >= 2 : 
            return False 
        
        # Same length --> move index sequentially 
        elif abs( len(s1) - len(s2) ) == 0:
            
            if s1[p1] != s2[p2]:
                count += 1 
            
            p1 += 1
            p2 += 1
            
        # Difference in Length == 1 
        else:
            
            if ( len(s1) - len(s2) ) > 0:    
                longer_str = s1
                shorter_str = s2 
            else:
                longer_str = s2
                shorter_str = s1
            
            if longer_str[p1] != shorter_str[p2]:
                count += 1
                p1 += 1
        
            p1 += 1
            p2 += 1
            
        # Regarding Count 
        if count > 1:
            return False 

    return True

one_differ(s1, s2)




# =========== 6. Assign Number in Minesqwwper 


import numpy as np

a = np.array([[-1,2,3],[3,4,5]])

ls = [[0,1],[0,2]]

b = np.pad(a, ((1,1),(1,1)), 'constant')

#np.pad(a, 1, 0)

def surround(arr, n, m ):
    
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            
            if i ==0 and j ==0:
                continue
            
            focal = arr[n + i, m +j ]
            if focal == -1:
                count += 1
                
    return count             
    
surround(b, 2, 2 )  


def mine_sweeper( bombs, num_row, num_col):
    
    # === Data Generation 
    arr = np.zeros([num_row, num_col])

    # Assign Bomb into the array 
    for item in bombs:
        arr[item[0], item[1] ] = -1 
        
    # === Padding 
    
    arr1 = np.pad(arr, ((1,1),(1,1)), 'constant')

    # === loop through  
    
    for row in range( 1, num_row + 1 ):
        for col in range( 1, num_col + 1 ):
            
            # Jump Bomb 
            if arr1[row, col] == -1: 
                continue
            
            # 
            else: 
                count = surround(arr1, row, col )
                arr1[row, col] = count

    # Remove Zero Padding 
    arr2 = arr1[ 1 : arr.shape[0] + 1 ,  1 : arr.shape[1] + 1,   ]

    return arr2

arr2 = mine_sweeper( [[0,0],[0,1]], 3, 4) 
# [[-1,-1,1,0],[2,2,1,0] ,[0,0,0,0] ]

# ====== 7. Find Where to expand in MineSweeper 














# ======== Daily Coding Problem 3 (Google)

class Node:
    
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

node = Node('root', Node('left', Node('left.left')), Node('right') ) 

assert deserialize(serialize(node)) 
left.left.val == 'left.left'



