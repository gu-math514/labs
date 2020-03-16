
# This was going to be hw6


# task 1 - write a python class that supports addition of numpy objects

import numpy as np
import itertools
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def atleast_2d(x,axis=1):
    return x if np.ndim(x) > 1 else np.expand_dims(x,axis)

# convert data to numpy array
def box_data(data):
    if type(data) is not np.ndarray:
        data = np.array(data)
        if data.ndim <= 1:
            return data.reshape((1,-1))
    return data

# a simple class that can 'add'
class CGNode1:
    def __init__(self,data,op=None):
        self.data = box_data(data)
        self.op = op  # operatrion that created this node
        
    def __add__(self,rhs):
        rdata = rhs.data if isinstance(rhs,CGNode1) else rhs
        return CGNode1(self.data+rdata,'+')
    
    def __radd__(self,lhs):
        return CGNode1(lhs+self.data,'+')
    
    # called when enter object name on command line
    def __repr__(self):
        return (self.op if self.op else "None")+":"+self.data.__repr__()
    
    def __str__(self):
        return (self.op if self.op else "None")+":"+self.data.__repr__()   
    

if False:
    b=CGNode1(1)
    print(b)
    
    foo=b+2.
    print(foo)
    
    bar=3+foo
    print(bar)
    
    x=1+2+3+4+CGNode1(5.)
    print(x)
 
# task 2 - write a python class that supports addition (like task1) and traces
#          operators (builds a computational graph)
    
# a simple class that can trace operations
class CGNode2:
    id_counter=itertools.count(1)
    def __init__(self,data,op=None,parents=[]):
        self.id = next(CGNode2.id_counter)
        self.data = box_data(data)
        self.op = op
        self.parents = parents
        
    def leaf(self): return self.op is None
    
    def dump(self,sorted=False):
        print(self.__str__())
        for p in self.parents: p.dump()
        
    def __add__(self,rhs):
        if isinstance(rhs,CGNode2):
            rdata = rhs.data
            return CGNode2(self.data+rdata,'+',[self,rhs])
        return CGNode2(self.data+rhs,'+',[self])
    
    def __radd__(self,lhs):
        return CGNode2(lhs+self.data,'+',[self])
    
    # called when enter object name on command line
    def __repr__(self):
        return (self.op if self.op else "None")+":"+str(self.id)+":"+self.data.__repr__()
    
    def __str__(self):
        return (self.op if self.op else "None")+":"+str(self.id)+":"+self.data.__repr__()      
 
    
if False:
    a=CGNode2(1.)
    b=CGNode2(2.)
    c=CGNode2(3.)
    foo=a+b+c
    print(foo)
    print(foo.dump())
    

# task 3: extend your class from task 2 to support automatic differention
#         you will need to topo-sort your graph
#         and implement a small number of vector-jacobian-products

def topo_sort(var):
    vars_seen = set()
    top_sort = []
    def top_sort_helper(vr):
        if (vr in vars_seen) or vr.is_leaf():
            pass
        else:
            vars_seen.add(vr)
            for pvar in vr.parents:
                top_sort_helper(pvar)
            top_sort.append(vr)    
    top_sort_helper(var)
    return top_sort 
    
     
def bprop_sigmoid(node):
    new=(node.data*(1.-node.data)*node.grad)
    node.parents[0].set_grad(new)
    
def bprop_mul(node):
    new = node.grad * node.parents[1].data
    node.parents[0].set_grad(new)
    new = node.grad * node.parents[0].data
    node.parents[1].set_grad(new)  
 
def bprop_add(node):
    for p in node.parents: p.set_grad(node.grad)
    
def bprop_sub(node):
    node.parents[0].set_grad(node.grad)
    if len(node.parents) > 1:
        node.parents[1].set_grad(-1.*node.grad)

# one parent on rhs of a - b
def bprop_sub1(node):
    node.parents[0].set_grad(-1.*node.grad)    
    
bpd = {}
bpd['+']=bprop_add
bpd['-']=bprop_sub
bpd['*']=bprop_mul
    
# a simple class that can trace operations and compute gradients
class CGNode3:
    id_counter=itertools.count(1)
    def __init__(self,data,op=None,parents=[],find_grad=False):
        self.id = next(CGNode3.id_counter)
        self.data = box_data(data)
        self.grad = None
        self.find_grad = find_grad
        self.op = op
        self.parents = parents
        self.bprop = None if op is None else bpd[op]
        
    def is_leaf(self): return self.op is None
    
    def dump(self,sorted=False):
        print(self.__str__())
        for p in self.parents: p.dump()
       
    def sigmoid(self):
        return CGNode3(1./(1.+np.exp(-self.data)),"sig",[self])
    
    def set_grad(self,grad):
        #if self.find_grad:
        self.grad = grad if self.grad is None else self.grad + grad
            
    def backward(self,grad=None,verbose=False):
        if(grad is None):
            grad = np.ones_like(self.data)         
        # self should be last in topo-sort
        tsorted = topo_sort(self)
        self.grad = grad
        if verbose: print('root id=',self.id)
        for node in reversed(tsorted):
            if verbose: print('node id=',node.id,node.opc())
            node.bprop(node)
            
    def __add__(self,rhs):
        if isinstance(rhs,CGNode3):
            return CGNode3(self.data+rhs.data,'+',[self,rhs])
        return CGNode3(self.data+rhs,'+',[self])
    
    def __radd__(self,lhs):
        return CGNode3(lhs+self.data,'+',[self])
    
    def __sub__(self,rhs):
        if isinstance(rhs,CGNode3):
            return CGNode3(self.data-rhs.data,'-',[self,rhs])
        return CGNode3(self.data-rhs,'-',[self])
    
    def __rsub__(self,lhs):
        return CGNode3(lhs-self.data,'-',[self])
    
    def __mul__(self, rhs):   
        if isinstance(rhs,CGNode3):
            return CGNode3(self.data * rhs.data,'*',[self,rhs])
        new = CGNode3(rhs)   # need these values from backprop
        return CGNode3(self.data * rhs,'*',[self,new])
    
    def __matmul__(self, rhs):
        return CGNode3(self.data @ rhs.data,'@',[self,rhs])
    
    # called when enter object name on command line
    def __repr__(self):
        return (self.op if self.op else "None")+":"+str(self.id)+":"+self.data.__repr__()
    
    def __str__(self,show_data=False):
        if show_data:
            return (self.op if self.op else "None")+":"+str(self.id)+":"+self.data.__repr__()  
        return (self.op if self.op else "None")+":"+str(self.id)

if False:
    x=CGNode3(.5)
    y=x*(x-1.)*(x+1.)
    y.dump()
    y.data
    y.backward()
    
    def cubic(x):
        x=CGNode3(x)
        return x,x*(x-1.)*(x+1.)
    
    grid=np.linspace(-1.5,1.5,51)
    x,y=cubic(grid)
    y.backward()
    plt.plot(x.data,y.data,linewidth=3,label="$y(x)$")
    plt.plot(x.data,x.grad,linewidth=3,label="$y^\prime(x)$")
    plt.axhline(0,linewidth=2,color="gray")
    plt.legend()
    plt.show()
    

if False:
    b=CGNode3(np.zeros(3).reshape(1,-1))
    x=CGNode3(np.random.uniform(-1.,1.,size=(10,2)))
    w=CGNode3(np.random.randn(2,3))
    z=b+x@w
    y=z.sigmoid()


def bprop_add(node):
    for p in node.parents: p.set_grad(node.grad)
    
def bprop_sub(node):
    node.parents[0].set_grad(node.grad)
    if len(node.parents) > 1:
        node.parents[1].set_grad(-1.*node.grad)

# one parent on rhs of a - b
def bprop_sub1(node):
    node.parents[0].set_grad(-1.*node.grad)
    
def bprop_mul(node):
    new = node.grad * node.parents[1].data
    node.parents[0].set_grad(new)
    new = node.grad * node.parents[0].data
    node.parents[1].set_grad(new)  
    
def bprop_div(node):
    a = node.parents[0].data
    b = node.parents[1].data
    new = node.grad / b
    node.parents[0].set_grad(new)
    new = -node.grad * a/(b*b)
    node.parents[1].set_grad(new) 
    
def bprop_matmul(node):
    c0 = node.parents[0]
    c1 = node.parents[1]
    new = node.grad @ c1.data.T
    c0.set_grad(new)
    new = (node.grad.T @ c0.data).T
    c1.set_grad(new)

def bprop_tpose(node):
    new = node.grad.T
    node.parents[0].set_grad(new)
    
def bprop_sigmoid(node):
    new=(node.data*(1.-node.data)*node.grad)
    node.parents[0].set_grad(new)

def bprop_relu(node):
    new=node.grad.copy()
    new[node.parents[0].data<0]=0
    node.parents[0].set_grad(new)
    
def bprop_tanh(node):
    new=(1.-node.data*node.data)*node.grad
    node.parents[0].set_grad(new)

def bprop_log(node):
    new=np.exp(-1.*node.data)*node.grad
    node.parents[0].set_grad(new)
    
def bprop_exp(node):
    new=node.data*node.grad
    node.parents[0].set_grad(new)
    
def bprop_sum(node):
    m = node.data.shape[0]
    node.parents[0].set_grad(np.full_like(node.parents[0].data,node.grad.data))
    
def bprop_sum0(node):
    m = node.data.shape[0]
    node.parents[0].set_grad(np.tile(node.grad,(m,1)))
      
def bprop_sum1(node):
    m = node.data.shape[1]
    node.parents[0].set_grad(np.tile(node.grad,m))

def bprop_bcast0(node):
    node.parents[0].set_grad(np.sum(node.grad,0))
    
def bprop_bcast1(node):
    node.parents[0].set_grad(np.sum(node.grad,1))
    
def bprop_get(node):
    new_grad=np.zeros_like(node.parents[0].data)
    new_grad[np.where(node.parents[1].data),:]=node.grad
    node.parents[0].set_grad(new_grad)

def bprop_eq(node):
    assert False, 'nprop_eq not implemented'
  
opd = {}

opd[bprop_add.__name__]='+'
opd[bprop_sub.__name__]='-'
opd[bprop_sub1.__name__]='r-'
opd[bprop_mul.__name__]='*'
opd[bprop_div.__name__]='/'
opd[bprop_matmul.__name__]='@'
opd[bprop_tpose.__name__]='T'
opd[bprop_sigmoid.__name__]='sig'
opd[bprop_relu.__name__]='relu'
opd[bprop_tanh.__name__]='tanh'
opd[bprop_log.__name__]='log'
opd[bprop_sum.__name__]='sum'
opd[bprop_sum0.__name__]='sum0'
opd[bprop_sum1.__name__]='sum1'
opd[bprop_bcast0.__name__]='bc0'
opd[bprop_bcast1.__name__]='bc1'
opd[bprop_exp.__name__]='**' 
opd[bprop_get.__name__]='[]' 
#opd[bprop_eq.__name__]='==' 

def check_dims(a,b):
    return np.all(a.shape==b.shape) 

class CGNode:
    id_counter=itertools.count(1)
    def __init__(self,data,parents=[],is_leaf=True,bprop=None):
        self.id = next(CGNode.id_counter)
        self.data = self.box_data(data)
        self.parents = parents
        self.leaf = is_leaf
        self.bprop = bprop
        self.grad = None
        
    def is_leaf(self): return self.leaf
        
    def box_data(self,data):
        if type(data) is not np.ndarray:
            data = np.array(data)
            #if data.size == 1:
            #    return data.reshape((1,1))
            if data.ndim <= 1:
                return data.reshape((1,-1))
        return data
    
    # need to make broadcasting explicit
    # many cases are not covered
    def __add__(self,rhs):
        if isinstance(rhs,CGNode):
            if self.data.shape[0] < rhs.data.shape[0]:  # insert broadcast node
                new=CGNode(np.broadcast_to(self.data,rhs.data.shape),[self],is_leaf=False,bprop=bprop_bcast0)
                #print('add, broadcast',new.id)
                assert check_dims(new.data,rhs.data), 'add: shape problem'
                return CGNode(new.data+rhs.data,[new,rhs],is_leaf=False,bprop=bprop_add)
            assert check_dims(self.data,rhs.data), 'add: shape problem'
            return CGNode(self.data+rhs.data,[self,rhs],is_leaf=False,bprop=bprop_add)
        return CGNode(self.data+rhs,[self],is_leaf=False,bprop=bprop_add)
    
    def __radd__(self,lhs):
        new= CGNode(lhs+self.data,[self],is_leaf=False,bprop=bprop_add)
        #print('radd',new.id)
        return new
        
    def __sub__(self,rhs):
        if isinstance(rhs,CGNode):
            return CGNode(self.data-rhs.data,[self,rhs],is_leaf=False,bprop=bprop_sub)
        return CGNode(self.data-rhs,[self],is_leaf=False,bprop=bprop_sub)
    
    def __rsub__(self,lhs):
        return CGNode(lhs-self.data,[self],is_leaf=False,bprop=bprop_sub1)
    
    def __mul__(self, rhs):   
        if isinstance(rhs,CGNode):
            return CGNode(self.data * rhs.data,[self,rhs],is_leaf=False,bprop=bprop_mul)
        new = CGNode(rhs)
        return CGNode(self.data * rhs,[self,new],is_leaf=False,bprop=bprop_mul)
    
    def __rmul__(self, lhs):
        new = CGNode(lhs)
        #print('rmul',new.id)
        return CGNode(lhs * self.data,[new,self],is_leaf=False,bprop=bprop_mul)
    
    def __truediv__(self,rhs):
        if isinstance(rhs,CGNode):
            return CGNode(self.data / rhs.data,[self,rhs],is_leaf=False,bprop=bprop_div)
        new = CGNode(rhs)
        return CGNode(self.data / rhs,[self,new],is_leaf=False,bprop=bprop_div)
    
    def __matmul__(self, rhs):
        return CGNode(self.data @ rhs.data,[self,rhs],is_leaf=False,bprop=bprop_matmul)
    
    def __rmatmul__(self, lhs):
        new=CGNode(lhs)
        print('rmatmul',lhs.shape,self.data.shape)
        return CGNode(lhs @ self.data,[new,self],is_leaf=False,bprop=bprop_matmul)
    
    # if op '==' overloaded, then need to overload hash so can use in dict -- so avoiding that
    def equal(self,rhs):
        iseq = self.data == rhs.data if isinstance(rhs,CGNode) else self.data == rhs
        return CGNode(iseq)  # no backprop through logical check

    def __getitem__(self,i):
        if isinstance(i,CGNode):
            return CGNode(self.data[i.data.squeeze()],[self,i],is_leaf=False,bprop=bprop_get)
        new = CGNode(i)
        return CGNode(self.data[i.squeeze()],[self,new],is_leaf=False,bprop=bprop_get)

    def transpose(self):
        return CGNode(self.data.transpose(),[self],is_leaf=False,bprop=bprop_tpose)
    
    def sigmoid(self):
        return CGNode(1./(1.+np.exp(-self.data)),[self],is_leaf=False,bprop=bprop_sigmoid)
    
    def relu(self):
        return CGNode(np.maximum(self.data, 0),[self],is_leaf=False,bprop=bprop_relu)
        
    def tanh(self):
        return CGNode(np.tanh(self.data),[self],is_leaf=False,bprop=bprop_tanh)
    
    def log(self):
        return CGNode(np.log(self.data),[self],is_leaf=False,bprop=bprop_log)
      
    def exp(self):
        return CGNode(np.exp(self.data),[self],is_leaf=False,bprop=bprop_exp)

    def sum(self,dim=0):
        if dim is None:
            return CGNode(np.sum(self.data),[self],is_leaf=False,bprop=bprop_sum)
        if dim==0:
            return CGNode(np.sum(self.data,0),[self],is_leaf=False,bprop=bprop_sum0)
        return CGNode(np.sum(self.data,1),[self],is_leaf=False,bprop=bprop_sum1)
    
    def bcast(self,dim,count):
        if dim==0:
            return CGNode(np.tile(self.data,(count,1)),[self],is_leaf=False,bprop=bprop_bcast0)
        return CGNode(np.tile(self.data,count),[self],is_leaf=False,bprop=bprop_bcast1)
    # with this approach, topo-sort controls order, so just do update for node
    def set_grad(self,grad=None):
        self.grad = grad if self.grad is None else self.grad + grad
    
    def backward(self,grad=None,verbose=False):
        if(grad is None):
            grad = np.ones_like(self.data)         
        # self should be last in topo-sort
        tsorted = topo_sort(self)
        self.grad = grad
        if verbose: print('root id=',self.id)
        for node in reversed(tsorted):
            if verbose: print('node id=',node.id,node.opc())
            node.bprop(node)
    
    def step(self,lr):
        self.data -= lr*self.grad
    
    # could set to None
    def zero_grad(self):
        self.grad = np.zeros(self.data.shape)
    
    def dump(self,sorted=False):
        if not sorted:
            print(self.__str__())
            for p in self.parents: p.dump()
            return
        tsorted = topo_sort(self)
        tsorted[-1].dump()
        
    def opc(self):
        opc = 'noop'
        if self.bprop is not None:
            opc = opd[self.bprop.__name__] if self.bprop.__name__ in opd.keys() else 'noop'
        return opc
    
    def __repr__(self):
        bprop = 'None' if self.bprop is None else str(self.bprop).split(' ')[1]
        return self.data.__repr__()+bprop
    def __str__(self):
        return f'[id:{self.id},op:{self.opc()},pids:{list(map(lambda a:a.id,self.parents))},is_leaf:{self.is_leaf()}]'
    
if False:
    # To Do -- check numerical gradients of net with tanh and relu
    
    # test 1 -- check numerical gradients
    from collections import deque
    
    x=np.array([[1.,2.]])
    w=np.array([3.,4.]).reshape(2,1)
    b=-1.
    def f(): return -1*np.log(sigmoid(b+x@w))
    params=[x,w]
    grad=deque()
    eps=1.e-6
    for p in params:
        g = np.zeros_like(p)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                save = p[i,j]
                p[i,j]= save + eps
                cp=f()
                p[i,j]= save - eps
                cm=f()
                p[i,j] = save
                g[i,j]=(cp-cm)/(2.*eps)
        grad.append(g)

    x=np.array([[1.,2.]])
    w=np.array([3.,4.]).reshape(2,1)
    xf=CGNode(x)
    wf=CGNode(w)
    #xf @ wf
    zf=b+xf @ wf
    y=zf.sigmoid()
    #y.backward()
    ylog = y.log()
    t=CGNode(1.)
    nt = -1*t
    cost = nt*ylog
    cost.dump()
    cost.backward()
    print(np.amax(np.abs(xf.grad-grad[0])))
    print(np.amax(np.abs(wf.grad-grad[1])))
    

# test 2 -- logistic regression
class agrad_logistic:
    def __init__(self,b,w):
        self.b = b
        self.w = w
        self.params = [b,w]
        
    def forward(self,x):         
        z = (self.b + x @ self.w)
        y = z.sigmoid()
        return z,y
    
    # this assumes certain relationship between cost function and output function
    def backward(self,x,t):
        z,y=self.forward(x)
        z.backward(y.data-t)
        
    def train(self,x,t,lr,epochs):
        batch_size = x.data.shape[0]
        for epoch in range(1,epochs+1):
            self.backward(x,t)
            for p in self.params: 
                p.step(lr/batch_size)
                p.zero_grad()
            y = self.fit(x)
            print(epoch,self.cost(y,t),self.accuracy(y,t))
    # call with fit-values
    def cost(self,y,t):
        m=y.data.size
        return -(np.sum(np.log(y.data[t==1]))+np.sum(np.log(1-y.data[t==0])))/m
    def fit(self,x):
        _,y = self.forward(x)
        return y
    # call with fit-values
    def accuracy(self,y,t):
        #z,y=self.forward(x)
        pred= 1*(np.squeeze(y.data)>=0.5)
        return np.sum(1*(t.squeeze()==pred)/len(pred))
    def __call__(self,x):
        return self.fit(x)

def mserror(y,t):
    m = t.size if type(t) is np.ndarray else t.data.size
    e=y-t
    return .5*(e*e).sum()/m

# won't work until subsetting supported
def nllerror(y,t):
    m = t.size if type(t) is np.ndarray else t.data.size
    return -1.*(y[t==1].log().sum()+(1.-y[t==0]).log().sum())/m

class agrad_logistic_v2:
    def __init__(self,b,w,costf):
        self.b = b
        self.w = w
        self.params = [b,w]
        self.costf = costf
        
    def forward(self,x):         
        z = (self.b + x @ self.w)
        y = z.sigmoid()
        return z,y
        
    def train(self,x,t,lr,epochs):
        batch_size = x.data.shape[0]
        for epoch in range(1,epochs+1):
            #self.backward(x,t)
            _,y = self.forward(x)
            cost = self.costf(y,t)+.5*(self.w.data*self.w.data).sum()/batch_size
            cost.backward()
            for p in self.params: 
                p.step(lr/batch_size)
                p.zero_grad()
            y = self.fit(x)
            print(epoch,self.cost(y,t),self.accuracy(y,t))
    def cost(self,y,t):
        return self.costf(y,t)
    def fit(self,x):
        _,y = self.forward(x)
        return y
    # call with fit-values
    def accuracy(self,y,t):
        #z,y=self.forward(x)
        pred= 1*(np.squeeze(y.data)>=0.5)
        return np.sum(1*(t.squeeze()==pred)/len(pred))    
    def __call__(self,x):
        return self.fit(x)

# call with tuple of (mean,cov) tuples
from collections.abc import Iterable
from collections import deque
def gmix_nd(args):
    if not len(args): assert False, 'Empty tuple'
    tup0=args[0]
    nd = len(tup0[0]) if isinstance(tup0[0],Iterable) else 1
    print(nd)
    def f(counts):
        d = deque()
        print(len(args),len(counts))
        for class_num,(n,(m,c)) in enumerate(zip(counts,args)):
            if nd>1 and len(m) != nd: assert False, 'Mismatched dimensions'
            xy = np.random.multivariate_normal(m,c,n) if nd>1 else np.random.normal(m,c,n)
            xyt = np.column_stack((xy,np.repeat(class_num,n)))
            d.append(xyt)
        data=np.vstack(d)
        np.random.shuffle(data)
        return data[:,:nd], (data[:,nd]).astype(int)
    return f

if False: # test div
    x=CGNode(np.array(np.arange(5)).reshape(-1,1))
    foo=.5*(x*x).sum()/x.data.size

if False:
    def to_numpy(x):
        return x if type(x) is np.ndarray else x.data
    
    gmixf=gmix_nd(((np.array([1,1]),np.diag([1,1])),\
                   (np.array([3,3]),np.diag([1,1]))))
    nx=51; ny=51
    xg=np.linspace(-2.,6.,nx)
    yg=np.linspace(-2.,6.,ny)
    grid = np.column_stack((np.repeat(xg,ny),np.tile(yg,nx)))
    colors = np.array(['g','b','r'])

    xy,t=gmixf((60,40))
    t=atleast_2d(t)
    inp_d = xy.shape[1]
    out_d = 1
    
    
    lr = .15
    b = CGNode(0.)
    w = CGNode(np.random.rand(inp_d,out_d)*.01)
    x = CGNode(xy)   # rmatmul doesn't seem to work
    model=agrad_logistic(b,w)
    model.train(x,t,lr,100)
    pred = 1*(to_numpy(model(CGNode(grid))).squeeze() > 0.5)
    plt.scatter(grid[:,0],grid[:,1],c=colors[pred],alpha=.3)
    pred = 1*(to_numpy(model(x)).squeeze() > 0.5)
    pred[pred!=t.squeeze()]=2
    plt.scatter(xy[:,0],xy[:,1],c=colors[pred])
    plt.show()
    
    b = CGNode(0.)
    w = CGNode(np.random.rand(inp_d,out_d)*.01)
    model=agrad_logistic_v2(b,w,mserror)
    model.train(x,t,lr,100)
    
        
    b = CGNode(0.)
    w = CGNode(np.random.rand(inp_d,out_d)*.01)
    model=agrad_logistic_v2(b,w,nllerror)
    model.train(x,t,lr,100)
