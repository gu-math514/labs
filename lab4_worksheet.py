
import numpy as np
from collections import deque
from collections.abc import Iterable
import matplotlib.pyplot as plt

# exercise 0: Questions about hw3?

# exercise 1

# part a.
# write a function that ensures a numpy array-like object has 2 or more dimensions
# - if object already has 2/more dims, returned as is
# - if object has 1 dimension, a dimension of length 1 is added
# you might want to use numpy.ndim and numpy.expand_dims

# part b.
# what does numpy.atleast_2d do?
# test your function and the numpy function on 1d and 2d arrays






# exercise 2
# write a function to return a one-hot encoding matrix given a set of integer values.
# make the one-hot vectors rows in the returned matrix






# exercise 3
# given a vector of real numbers, compute the stables softmax matrix
# (show softmax slide)




# exercise 4
# write a function to compute a confusion table given two integer vectors



# exercise 5
# part a. do a scatter plot of the following 3-class data.
#         use colors green,blue,cyan for classes 0,1,2
 

def gmix_ndv2(*args):
    def f(*counts):
        d = deque()
        for class_num,(n,(m,c)) in enumerate(zip(counts,args)):
            #print(class_num,n,m,c)
            nd = len(m) if isinstance(m,Iterable) else 1
            xy = np.random.multivariate_normal(m,c,n) if nd>1 else np.random.normal(m,c,n)
            xyt = np.column_stack((xy,np.repeat(class_num,n)))
            d.append(xyt)
        data=np.vstack(d)
        np.random.shuffle(data)
        return data[:,:nd], (data[:,nd]).astype(int)
    return f

set1 = (np.array([.5,.5]),np.diag([.2,2]))
set2 = (np.array([1.5,1.5]),np.diag([.1,2]))
set3 = (np.array([1.5,0]),np.diag([.1,2]))
gmixf = gmix_ndv2(set1,set2,set3)

xy,t=gmixf(50,40,30)


# exercise 5 part b.  write code that demonstrates the decision boundaries
# for the "predictor" code below.
# first, look at the provided functions xy_data_range, xy_grid_pts
# can you describe what they do?
# second, look at the 'predict' function. Can you describe what it does?
# note that 'predict' uses the stable_softmax function
# the code randomly chooses values for b,w. why are those dimensions correct?
# what could I have done to make this code more generic?

# finally, use this code and the random b,w vectors and plot the
# resulting decision boundaries and the predictions for the data set
# if a data point is predicted incorrectly, color it red

# utility function - compute min/max values for xy data
def xy_data_range(xy):
    xmin,ymin = np.amin(xy,axis=0)
    xmax,ymax = np.amax(xy,axis=0)
    return (xmin,xmax), (ymin,ymax)
# utility function - given x-grid and y-grid, compute xy data covering a rectangle
# xpts = np.linspace(xmin-margin,xmax+margin,NX)
def xy_grid_pts(xpts,ypts):
    nx = len(xpts); ny = len(ypts)
    return np.column_stack((np.repeat(xpts,ny),np.tile(ypts,nx)))

b=np.random.uniform(size=3).reshape(1,3)
w=np.random.uniform(-1,1,6).reshape(2,3)
def predictf(b,w):
    def predict(x):
        z = b+x@w
        y = stable_softmax(z)
        c = np.argmax(y,axis=1)
        return c
    return predict
predict=predictf(b,w)














# exercise 1

# write a function that ensures a numpy array-like object has 2 or more dimensions
# - if object already has 2/more dims, returned as is
# - if object has 1 dimension, a dimension of length 1 is added
# you might want to use numpy.ndim and numpy.expand_dims

def atleast_2d(x,axis=1):
    return x if np.ndim(x) > 1 else np.expand_dims(x,axis)

# what does numpy.atleast_2d do?
x=np.random.uniform(size=3)
print(x.shape)
print(atleast_2d(x).shape)
print(np.atleast_2d(x).shape)
    
# exercise 2
# write a function to return a one-hot encoding matrix given a set of integer values.
# make the one-hot vectors rows in the returned matrix

def one_hot_simple(y,axis=1):
    z = y.squeeze()
    z = z-np.amin(z)
    nc=np.amax(z)+1
    return np.eye(nc,dtype=int)[z] if axis==1 else np.transpose(np.eye(nc,dtype=int)[z])

y=np.random.randint(0,5,size=20)
print(y,one_hot_simple(y))


# exercise 3
# given a vector of real numbers, compute the softmax matrix
    
# axis=0 if normalizing columns
# would be better if max was over rows/cols
def stable_softmax(a, axis=1):
    aa=np.exp(a-np.amax(a))
    return aa/np.sum(aa,axis) if axis==0 else (aa.T/np.sum(aa,1)).T

a=np.random.uniform(size=12).reshape(4,3)
print(a)
b=stable_softmax(a)
print(b)
print(np.sum(b,axis=1))


# exercise 4
# write a function to compute a confusion table given two integer vectors


def confusion_tab(truth,predict):
    nc=np.max((len(np.unique(truth)),len(np.unique(predict))))
    counts=np.zeros((nc,nc),int)
    min_truth=truth.min()
    for i in range(len(truth)):
        counts[truth[i]-min_truth,predict[i]-min_truth]+=1
    return counts

truth=np.random.randint(low=0,high=6,size=20)
predict=truth
print(confusion_tab(truth,predict))
print(np.sum(1*(truth==predict)))
predict=np.random.randint(low=0,high=6,size=20)
print(confusion_tab(truth,predict))
print(np.sum(1*(truth==predict)))



# exercise 5

def gmix_ndv2(*args):
    def f(*counts):
        d = deque()
        for class_num,(n,(m,c)) in enumerate(zip(counts,args)):
            #print(class_num,n,m,c)
            nd = len(m) if isinstance(m,Iterable) else 1
            xy = np.random.multivariate_normal(m,c,n) if nd>1 else np.random.normal(m,c,n)
            xyt = np.column_stack((xy,np.repeat(class_num,n)))
            d.append(xyt)
        data=np.vstack(d)
        np.random.shuffle(data)
        return data[:,:nd], (data[:,nd]).astype(int)
    return f

set1 = (np.array([.5,.5]),np.diag([.2,2]))
set2 = (np.array([1.5,1.5]),np.diag([.1,2]))
set3 = (np.array([1.5,0]),np.diag([.1,2]))
gmixf = gmix_ndv2(set1,set2,set3)

xy,t=gmixf(50,40,30)

colors =  np.array(['g','b','c'])
plt.scatter(xy[:,0],xy[:,1],c=colors[t])

# assumes x,y data in rows
def xy_data_range(xy):
    xmin,ymin = np.amin(xy,axis=0)
    xmax,ymax = np.amax(xy,axis=0)
    return (xmin,xmax), (ymin,ymax)
# xpts = np.linspace(xmin-margin,xmax+margin,NX)
def xy_grid_pts(xpts,ypts):
    nx = len(xpts); ny = len(ypts)
    return np.column_stack((np.repeat(xpts,ny),np.tile(ypts,nx)))

b=np.random.uniform(size=3).reshape(1,3)
w=np.random.uniform(-1,1,6).reshape(2,3)
def predictf(b,w):
    def predict(x):
        z = b+x@w
        y = stable_softmax(z)
        c = np.argmax(y,axis=1)
        return c
    return predict
predict=predictf(b,w)

xrng,yrng = xy_data_range(xy)
grid = xy_grid_pts(np.linspace(xrng[0],xrng[1],101),np.linspace(yrng[0],yrng[1],101))
grid_pred = predict(grid).astype(int)

plt.figure(figsize=(10,7))
plt.scatter(grid[:,0],grid[:,1],s=5,c=colors[grid_pred])
plt.scatter(xy[:,0],xy[:,1],c=colors[t])
plt.axhline(y=0)
plt.axvline(0)
pred=predict(xy)
bads=np.where(pred != t)
plt.scatter(xy[bads,0],xy[bads,1],c='r')
plt.show()







