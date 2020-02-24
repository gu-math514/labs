

import numpy as np
import matplotlib.pyplot as plt

# 1. Activation functions
# Write 3 closures that accept needed arguments and return f(x), df/dx(x) for following 3 closures
# a. relu,  f(z)=max(0,z)
# b. selu,  f(z)=lambda*z for z>=0, =lambda*alpha*(e^z-1) for z<0
# c. swish, f(z)=z*sigmoid(beta*z)

# write code to numerically check your activation function derivatives


# 2. Generators/Data Loaders
# The following code is an example of a generator
# What do you get when you run print(fib(1))?
# Use fib(50) in an enumeration loop and print the index and value. 
# What is the largest fibonacci number less that 50?
def fib(n):
    a,b=0,1
    while a<n:
        yield a
        a,b=b,a+b

# Want to use this idea to generate batches of data for model training.
# As a next step, write a closure that returns a generator. The generator should return
# arrays of integers of length batch_size which are subsets of 1,...,n. For now
# those integers can be in order
# Decide what to do if batch_size does not divide n evenly
# 1. call your generator in a loop and print the index and the returned data
#    exhausting the generator completes one epoch of data
# 2. wrap your generator loop in an epoch loop. Again, print the index and returned data
#    you should see the same data in the same order for each epoch


def data_genf(n,batch_size):
    def generator():
        pass
    return generator

# Add the argument 'shuffle=True' to your closure. When shuffle is true have the generator shuffle 
# the data so different samples are grouped into batches with each epoch

def data_genf(n,batch_size,shuffle=True):
    def generator():
        pass
    return generator    
        
# okay, now want to use our data generator to serve up batches of data
# assume our data is x,y where x is an array with samples arranged in rows
# y are the tags
# write a third version of your closure/generator that now takes x,y as inputs and
# it returns shuffled batches x,y on each call.
# use the following for x,y
        
# simple data loader for small array data sets
def array_data_genf(x,y,batch_size=None,shuffle=True):
    def generator():        
        pass
    return generator  


 
# 3. Learning rate schedulers
# populate the following signatures to implement a linear rate schedule that goes
# from lr_start to lr_final over n calls to step
# the rate should stay at lr_final if step called more than nsteps times
# plot the rates to test your code
        
class LRSched:
    def __init__(self,lr_start,lr_final,nsteps):
        pass
        
    def step(self):
        pass
        
    def get_lr(self):
        pass  






    
    
    
    
    
    
    
    
    
    
    
    
    


# 1. Activation functions

def sigmoid(x):
    return 1/(1+np.exp(-x))

def reluf():
    def relu(z):
        return np.maximum(0.,z)
    def drelu(z):
        return np.greater(z,0).astype(int)
    return relu,drelu

def selu(x,alpha=1.67,lambda_=1.05):
    return lambda_*np.where(x < 0, alpha * (np.exp(x) - 1), x)
def dselu(x,alpha=1.67,lambda_=1.05):
    return np.where(x < 0, lambda_*alpha*np.exp(x), lambda_)
def seluf(alpha=1.67,lambda_=1.0507):
    return lambda z: selu(z,alpha,lambda_), lambda z: dselu(z,alpha,lambda_)

def swish(z,beta=1):
    return z*sigmoid(beta*z)
def dswish(z,beta=1):
    sig=sigmoid(beta*z)
    return sig+beta*z*sig*(1-sig)
def swishf(beta=1.):
    return lambda z: swish(z,beta=beta), lambda z: dswish(z,beta=beta)

# Note: looking at gradient differences point-by-point shows they
#       agree everywhere except at zero -- why?
for closuref in [reluf,seluf,swishf]:
    actf,actfp=closuref()
    xg=np.linspace(-10.,10.,21)
    eps=1.e-6
    for x in xg:
        print(x,actfp(x),(actf(x+eps)-actf(x-eps))/(2.*eps))
    print("max abs diff")
    print(np.max(np.abs(actfp(xg)-(actf(xg+eps)-actf(xg-eps))/(2.*eps))))


# 2. Generators/Data Loaders
   
for i,y in enumerate(fib(50)):
    print(i,y) 
    
def data_genf(n,batch_size=None):
    def generator():
        nb=1 if batch_size is None else max(1,n//batch_size)
        data_set = [i for i in range(n)]
        for batch in range(nb):
            if batch_size is None:
                samples = data_set
            else:
                offset=batch*batch_size
                samples=data_set[offset:offset+batch_size]
            yield samples
    return generator

data_gen=data_genf(12,5)
for epoch in range(3):
    for batch_num,batch_data in enumerate(data_gen()):
        print(batch_num,batch_data)


def data_genf(n,batch_size=None,shuffle=True):
    def generator():
        nb=1 if batch_size is None else max(1,n//batch_size)
        data_set = [i for i in range(n)]
        if shuffle: np.random.shuffle(data_set)
        for batch in range(nb):
            if batch_size is None:
                samples = data_set
            else:
                offset=batch*batch_size
                samples=data_set[offset:offset+batch_size]
            yield samples
    return generator

data_gen=data_genf(12,5)
for epoch in range(3):
    for batch_num,batch_data in enumerate(data_gen()):
        print(batch_num,batch_data)

# simple data loader for toy array data
def array_data_genf(x,y,batch_size=None,shuffle=True):
    def generator():
        m=x.shape[0] 
        nb=1 if batch_size is None else max(1,m//batch_size)
        
        index = [i for i in range(m)]
        if shuffle: np.random.shuffle(index)
        for batch in range(nb):
            if batch_size is None:
                samples = index
            else:
                offset=batch*batch_size
                samples=index[offset:offset+batch_size]
            yield x[samples,:],y[samples] 
    return generator

x=np.array(np.arange(24)).reshape(-1,3)
y=np.sum(x,axis=1)

data_gen=array_data_genf(x,y,3)
for epoch in range(3):
    for batch_num,batch_data in enumerate(data_gen()):
        print(batch_num,batch_data)    
    
# 3. Learning Rate Schedulers
    
class LRSched:
    def __init__(self,lr_start,lr_final,nsteps):
        self.lr=lr_start
        self.lr_start = lr_start
        self.lr_final = lr_final
        self.nsteps = nsteps
        self.dlr = self.lr - self.lr_final
        
    def step(self):
        self.lr = max(self.lr_final,self.lr-self.dlr/(self.nsteps-1))
        
    def get_lr(self):
        return self.lr    

rates=[]
linear_sched=LRSched(.025,.001,100)
for epoch in range(120):
    rates.append(linear_sched.get_lr())
    linear_sched.step()

print(max(rates))
print(min(rates))
plt.plot(rates)



