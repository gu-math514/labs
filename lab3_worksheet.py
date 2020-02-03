
# lab 3 -- preparation for hw3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from collections import deque

# 1. Contour plots
# As preparation for countour plotting, look up the numpy.meshgrid function
# Use meshgrid to create a grid of points and plot the grid


# contour plotting in python uses mesh grids. Define a function(x,y) that returns sin(x)*cos(y)
# Look up the matplotlib function contourf. Use contourf and your meshgrid to plot
# a contour of sin*cos







# 2. Classes
# Implement a class that computes a quadratic form in n-dimensions
# Since your class implements a function, use the __call__ magic method to make it callable
# y = x.T @(A @ x) + b.T @ x +c
# The class has 3 parameters - a,b,c. Use a Parameter class to hold the parameter value and grad
# Here's a skeleton

class Parameter:
    def __init__(self,data,grad=None,name=None):
        pass

class QForm:
    def __init__(self,a,b,c):
        pass
    
    def parameters(self):   # return an iterable of the parameters
        pass

    def __call__(self,x):   # compute y given x
        pass
    
    def grad(self,x):       # compute the gradient dy/dx = (a.T + a) @ x + b
        pass
    
    def param_grad(self,x): # compute the three gradients of y wrt a,b,c
        pass                # dyda= x @ x.T,, dydb = x, dydc = 1.
        

# instantiate a QForm object with the following arguments. Make sure your parameters are 2d -- even if only 1x1
a = np.eye(2)
b = np.array([.5,-5.]).reshape(2,-1)
c = np.array(1.).reshape(1,1)
qform = QForm(a,b,c)

# 'call' your object with x=(0,0). You should get y=1.
# With same input call the grad and param_grad methods. 
# grad should return b
# param_grad should return 3 grads
# - verify that the a-gradient is a 2x2 matrix
# - verify that the b-gradient is a length 2 vector
# - verify that the c-gradient is the number 1.


# 3. Numerical Gradients
# Now want to compute numerical gradients of the quadratic form wrt the parameters a,b,c
# - It will be clear that using a separate class for parameters makes this task much easier
# We will break this into two steps
# - first, a function to compute numerical gradients (centered differences)
# - second, a function to compare the numerical gradients with the analytical gradients returned by qform.param_grad

def compute_num_grad(func,x,eps=1.e-6):
    pass

# this method should generate an output that structurally matches the output of param_grad
# You need to loop over each parameter of qform, and then loop over each element of the parameter
# This is a bit involved, so the code is at the bottom of the file -- but give it a shot first to see if you understand the ideas


# Now write a function to compare the two sets of gradients and return the maximum absolute difference

def check_gradients(func,x):
    pass

# this method calls func.param_grad to get the analytical gradients
# it then calls compute_num_grad(func,x) to get numerical estimates
# finally, it loops over each parameter checking gradient differences
# One solution is at the bottom of the file -- but think it through so you understand the idea
    


# Once compute_num_grad and check_gradients are working, use them to check the gradients on qform
x=np.random.uniform(size=2).reshape(2,-1)
check_gradients(qform,x)

# You should get something around 1.e-9

# 4. visualizing qform
# Let's finish by plotting the qform surfance (3d) and contours
# create a meshgrid for x in [-2.,2.] and y in [-2.,3.]  (this includes min point but otherwise not special)
# Compute qform values over this grid and generate a contour plot using something like
# note that since qform in not a function of (x,y), you need (I think) a loop to create values
plt.contour(xarray,yarray,values,levels=50)
plt.scatter(-.25,2.5,c='r')
plt.colorbar()
plt.show()
plt.contourf(xarray,yarray,values,levels=50)   # filled contour
plt.scatter(-.25,2.5,c='r')
plt.colorbar()
plt.show()

# Now use the same data and plot the qform surface in 3d using
ax=plt.axes(projection='3d')
ax.plot_surface(xarray,yarray,values)
plt.show()
















def compute_num_grad(func,x,eps=1e-6):
    grad=deque()
    for p in func.parameters():
        g = np.zeros_like(p.data)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                save = p.data[i,j]
                p.data[i,j]= save + eps
                cp=func(x)
                p.data[i,j]= save - eps
                cm=func(x)
                p.data[i,j] = save
                g[i,j]=(cp-cm)/(2.*eps)
        grad.append(g)
    return grad

def check_gradients(func,x):
    func.param_grad(x)
    ngrad = compute_num_grad(func,x)
    
    max_diff=0
    i=0
    for p in func.parameters():
        d = p.grad - ngrad[i]
        dd = np.sqrt(np.max(d*d))
        if dd > max_diff: max_diff = dd
        i+=1
    return max_diff
