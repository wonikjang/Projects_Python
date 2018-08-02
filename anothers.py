# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:12:04 2018

@author: 09732
"""

# Anothers 

# ==============  1. Character map to integer (a --> 1,... z --> 26 ) (Facebook)
# Input : Number, output : Possible number of message 
# Exception : 01 --> 0 , "" --> 1 
# O(n)
# Decoding From left to right 

# When it is inefficient --> (111111 , 6)  --> O(2^n) 


def num_ways(data):
    
    memo = [ None ] * ( len(data) + 1) 
    
    return helper( data, len(data), memo )

def helper(data, k, memo):
    # Base case
    if k ==0 : # num_ways("") = 1
         return 1  
     
    s = len(data) - k
    if data[s] == '0': # num_ways("011") = 0
        return 0 
    
    # memo applied
    if memo[k] != None:
        return memo[k]
    
    # Recursive case 
    result = helper(data, k - 1, memo)
    if k >=2 and int(data[s:s+2]) <= 26:
        result = helper(data, k - 2, memo)
    
    # memo saved    
    memo[k] = result 
    
    return result  



# ================= 2. Given coordinates, how to find k closest points to the origin ? (Amazon)
# = Find the K smallest elemetns in a given array of numbers 

# 1. Sorting  and get K first elements --> O(nlogn) using quick sort or merge sort 
# 2. Selection sort --> O(nk)
# 3. Max heap --> O(n + (n-k)logk ) 
    
# 3. What is heap?  Efficient way to keep track of the largest value in the given collection. 
# 1. create a
    




# ================ 3. Find the number of negative integers in a row-wise / column-wise sorted matrix (Amazon)
    

M = [ [-3,-2,-1,1], [-2,2,3,4],[4,5,7,8]]

# Ans : 4
# naive --> O(nm) , Optimal --> O(n+m)

def countNegInt(M, n, m):
    
    count = 0 
    i = 0 
    j = m - 1
    
    while j >= 0 and i < n:
        if M[i][j] < 0 :
            count += ( j + 1 )
            i += 1
        else: 
            j -= 1 
        
    return count 

countNegInt(M, 3, 4)   


    
# ================== 4.  Given an array of unique items, finnd all of its subsets (Facebook)
# [1,2] --> 2^2 = 4
# Represent subset with the same rank of original set --> {2} = [null, 2] is same rank of [1,2]


# Recursion 


def all_subsets(given_array):
    subset = [None] * len(given_array)
    return helper(given_array, subset, 0)
    

def print_set(subset):
    result_list = []
    for i in range(len(subset)):
        if subset[i] is None:
            continue
        else:
            result_list.append( subset[i] )
    print(result_list)
    

def helper(given_array, subset ,i ):

    if i == len(given_array):
        print_set(subset)

    else: 
        # Two Cases 
        # 1. Without --> Set as null & go to the next index 
        subset[i] = None
        helper(given_array, subset , i+1 )
        
        # 2. With 
        subset[i] = given_array[i]
        helper(given_array, subset , i+1 )


all_subsets( [1,2] )
        
# Iterative Solution 


# ================== 5. Lowest Common Ancestor (Microsoft)        

# 1. pathToX 


def pathToX(root, X):
    if root == None:
        return None
    if root.value == X :
        return stack(X)
    
    # Left side Check
    leftPath = pathToX(root.left, X)
    if leftPath is not None:
        leftPath.add(root.value)
        return leftPath
    
    # Right side Check 
    rightPath = pathToX(root.right, X)
    if rightPath is not None:
        rightPath.add(root.value)
        return rightPath
    
    return null


def LCA(root, j , k):
    pathToJ = pathToX(root, j)
    pathToK = pathToX(root, k)
    
    LCAToReturn = None
    while pathToJ and pathToK:
        jPop = pathToJ.pop() # pop the first added one 
        kPop = pathToK.pop() # pop the first added one 
        
        if jPop == kPop:
            LCAToReturn = jPop
        else: 
            break
        
   return LCAToReturn


 
        
        
        
# =============== 6.  Randomly order (Fisher- Yates shuffle, knuth shuffle )
   
# 1. Method 
# 1.1 Create an array and generate random number as key 
# 1.2 Sort in ascending order
# Time Complexity O(nlogn) - QuickSort case , space Complexity: O(n)
   
# 2. Method 
# 1.1 Pick 1 random number from n-1, and a queue with 1 space 
# 1.2 Swap n with 1 random number 
# Time Complexity : O(n) , Space Complexity : O(1)
   
import random
   
def redorder(arr):
    n = len(arr)
    
    for i in range( (n-1), 1-1, -1 ):
        pick = floor( (i+1) * random.uniform(0,1) )
        temp = arr[i]
        # Swap 
        arr[i]     = arr[pick]
        arrp[pick] = temp 
        
    return arr 




# ================= 7. Longest Consecutive Character 

# Input : 'AABCDDBBBEA' --> Output:  {B : 3}
    
def LCC( chars ):
    
    dict_cons = {}
    previous = None
    
    for char in chars: 
        
        # Add congiguous value 
        if previous == char: 
            dict_cons[char] += 1
        
        # Setting initial value 
        if char not in dict_cons: 
            dict_cons[char] = 1
            
        if previous != char and char in dict_cons:
            dict_cons[char] = 1
                        
        previous = char 
    
    return dict_cons

    
dict0 = LCC('AABCDDBBBEA')
    







        





    
    
    




