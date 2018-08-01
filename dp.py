#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 23:02:35 2018

@author: wonikJang
"""

# === Dynamic Programming 

# 1. Largest Common Subsequence 

import numpy as np


def LSC(P, Q, n, m):
    
#    arr = np.zeros([n + 1 , m + 1 ])
#    if arr is not None:
#        return arr[n][m]
    
    # ===== 
    if n == 0 or m ==0 :
        result = 0 
    elif P[n-1] == Q[m-1]:
        result = 1 + LSC(P, Q, n-1, m)
        
    elif P[n-1] != Q[m-1] :
        tmp1 = LSC(P,Q,n-1, m)
        tmp2 = LSC(P,Q,n,m-1)
        result = max(tmp1, tmp2)
        
#    arr[n][m] = result 
    
    return result
    
LSC('aa','aab',2,3)

# 1.1 Memoized 



def LSC(P, Q, n, m, arr):
    
#    arr = np.zeros([n + 1 , m + 1 ])
#    if arr is not None:
#        return arr[n][m]
    
    # ===== 
    if n == 0 or m ==0 :
        result = 0 
        
    elif P[n-1] == Q[m-1]:
        result = 1 + LSC(P, Q, n-1, m, arr)
        
    elif P[n-1] != Q[m-1] :
        tmp1 = LSC(P,Q,n-1, m, arr)
        tmp2 = LSC(P,Q,n,m-1, arr)
        result = max(tmp1, tmp2)
        

    arr[n][m] = result 
    print(arr)
    return result

    
res0 = LSC('aa','aab',2,3, np.zeros([3,4]))



# 2. Number of cases add up to Target 

# 2.1 Recursive 

def count_sets(arr, total) : 
    return rec(arr, total, len(arr)-1 )
    
    
def rec(arr, total, i):
    # Base Case 
    if total == 0 : 
        return 1 
    elif total < 0 :
        return 0
    elif i < 0 :
        return 0 

    # Recursive Case
    
    # Exception 
    elif total < arr[i]:
        return rec(arr, total, i-1)
    
    # 2 Cases : with the indexed or without the indexed 
    else:
        return rec(arr, total - arr[i], i-1) + rec(arr, total, i-1)
    

# 2.2 Memoized 
        
def count_sets_dp(arr, total) : 
    mem = {} 
    return dp(arr, total, len(arr)-1, mem )


def dp(arr, total, i, mem):
    
    # Memoized 
    key = str(total) + ":" + str(i)
    if key in mem:
        return mem[key]
    
    # Base Case 
    if total == 0 : 
        return 1 
    elif total < 0 :
        return 0
    elif i < 0 :
        return 0 
    elif total < arr[i]:
        to_return = dp(arr, total, i-1, mem)
    else: 
        to_return = ( dp(arr, total - arr[i], i-1, mem) + dp(arr, total, i-1, mem) )
    
    mem[key] = to_return

    return to_return    
        


count_sets_dp([2,4,6,10], 16)
