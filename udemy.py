#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 23:11:05 2018

@author: wonikJang
"""

# Udemy 

# Intro - find 3 elements that multiply to 100 

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






