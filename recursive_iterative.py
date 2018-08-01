
# ====================== Recursive vs. Iterative 

# Recursive way is more sophisticted way, but cost of calling the function itself and time CPU running time becomes longer 
#   - Problem: The larger number of recursion, the more methods are accumulated into the stack  , which encourages stack overflow 

# Iterative way requires a space for a variable list and a copy space, which can execute the method. 

# ============= Factorial 

# 1. Recursive way 

def factorial_recur(n):
    if n == 2 or n == 1:
        return n
    else:
        return n * factorial_recur(n-1)
    
# 2. Iterative way 
def factorial_iter(n):
    
    if n == 1 or n == 2:    
        return n
    
    else:
        result = 1
        while n > 1:
            result *= n
            n -= 1
            
        return result

factorial_iter(5)

# ======================== Back Propagation 


# ====== Why Non-convex problem? 
# Because activtion functions are non-linear and they are connected across layers, 
# weight optimization in neural network is non-convec optimization

# Therefore, in general case, figuring out global optimum of parameters in neural network is impossible. 
# This leads us to use gradient descent to find local optimum presumed to be close to global optimum 


# ====== How to update parameters? Back propagatioin 
# Basically, it's simplified chain rule of gradient descent. 
# With respect to optimization problem, target function should be defined and we can calculate loss function by comparing target output with estimated output 

# ====== Why use softmax loss in classification? 
# Because the gradient value in softmax loss is neumerically stable. 
# Normally, we can update parameter using such gradients. 

# ====== Then, why backpropagationi ? 
# Although we have to calculate the gradient w.r.t. current parameter, complex neural network makes it difficult. 
# So, back-propagation uses current parameter to caluculate loss, and how each parameter affected focal loss using chain rule, and finally update it.
# 2 procedures: 1. Propagation Phase 2. Weight Update Phase 
#   1. Propagation Phase: 
#       Forward Propagation : Calculate ouput from input training data and error from output node. (input --> hidden --> output)
#       Back Propagation    : Calculate how much nodes in the previous layer affected the error using error from output node  (output --> hidden)
#   2. Weight Update: 
#       Calculate gradinet of parameter using chain rule 





















































