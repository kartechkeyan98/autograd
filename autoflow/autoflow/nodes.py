import numpy as np
from collections import deque


class Node(np.ndarray):
    def __new__(
        subtype, shape, 
        dtype = float,
        buffer = None,
        offset = 0,
        strides = None,
        order = None
    ):
        newobj = np.ndarray.__new__(
            subtype, shape, dtype,
            buffer, offset, strides, order
        )

        # gradient related:
        newobj.grad = None          # local gradient at this node
        newobj.grad_fn = None       # grad fn(s) to backprop gradients
        newobj.noperands = 0        # no.of operands to create this
        newobj.operands = ()        # empty tuple
        newobj.opmethod = None      # computation operation represented by this unit
        return newobj
    

    def _nodify2(self, method, opname:str, other=None, self_first=True):
        if not isinstance(other, Node):
            other = Constant.create_node(other)
        # get a new ndarray with the results, beeyatch!
        opres = getattr(np.ndarray, method)(self,other)

        return Operation.create_node(
            opres, opname, 
            self if self_first else other,
            other if self_first else self
        )

    def __add__(self, other):
        # this only does the forward computation and returns the corr. operation node
        opres = self._nodify2(method='__add__', opname='add', other=other)
        # to take care of backward computation, we need a few things
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        # set the backward function    
        opres.grad_fn = addbackward
        return opres
    def __radd__(self, other):
        opres = self._nodify2('__radd__','add',other=other, self_first=False)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = addbackward
        return opres
    def __sub__(self,other):
        opres = self._nodify2('__sub__','sub',other=other)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = subbackward
        return opres
    def __rsub__(self,other):
        opres = self._nodify2('__rsub__','sub',other=other, self_first=False)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = subbackward
        return opres
    def __mul__(self,other):
        opres = self._nodify2('__mul__','mul',other=other)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = mulbackward
        return opres
    def __rmul__(self,other):
        opres = self._nodify2('__rmul__','mul',other=other,self_first=False)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = mulbackward
        return opres
    def __div__(self,other):
        opres = self._nodify2('__div__','div',other=other)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = divbackward
        return opres
    def __rdiv__(self,other):
        opres = self._nodify2('__div__','div',other=other, self_first=False)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = divbackward
        return opres
    def __truediv__(self,other):
        opres = self._nodify2('__truediv__','div',other=other)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = divbackward
        return opres
    def __rtruediv__(self,other):
        opres = self._nodify2('__rtruediv__','div',other=other,self_first=False)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = divbackward
        return opres
    def __pow__(self,other):
        opres = self._nodify2('__pow__','pow',other=other)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = powbackward
        return opres
    def __rpow__(self,other):
        opres = self._nodify2('__rpow__','pow',other=other,self_first=False)
        assert opres.noperands == 2, "No.of operands to the operation don't match up!"
        opres.grad_fn = powbackward
        return opres
    @property
    def T(self):
        '''
        augment np.ndarray T attribute
        '''
        opval = np.transpose(self)
        opres =  Operation.create_node(opval,'transpose',self)
        assert opres.noperands == 1, "No.of operands to the operation don't match up!"
        opres.grad_fn = transposebackward
        return opres
    
# different types of nodes in the fray

# 1. Variable Node
# purpose: store inputs and parameters which can be changed!
# in PyTorch parlance anything with requires_grad = True.
class Variable(Node):
    
    # static member variable to count unnmaed instances
    count_unnmaed = 0

    @staticmethod
    def create_node(val, name = None):
        '''
        This method creates a variable node and returns it
        Parameters:
        ----------
        val: number | np.ndarray
            it contains data we want, will be cvt to np.ndarray
            since we wanna use that buffer memory anyways
        name: str | None
            name of the node
        '''
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)
        obj = Variable(
            shape=val.shape,
            dtype=val.dtype,
            buffer=val,
            strides=val.strides
        )
        if name is not None:
            obj.name = name
        else:
            obj.name = "var_%d" %(Variable.count_unnmaed)
            Variable.count_unnmaed+=1
        
        # gradient related stuff
        obj.grad = np.zeros(obj.shape)

        return obj

# constant node
# it holds constant values, pretty simple as that
# it's grad will always be None
class Constant(Node):

    # static counter for unnamed instances
    count_unnamed = 0

    @staticmethod
    def create_node(val, name=None):
        '''
        This method creates a variable node and returns it
        Parameters:
        ----------
        val: number | np.ndarray
            it contains data we want, will be cvt to np.ndarray
            since we wanna use that buffer memory anyways
        name: str | None
            name of the node
        '''
        if not isinstance(val, np.ndarray):
            val = np.array(val, dtype=float)
        obj = Constant(
            shape=val.shape,
            dtype=val.dtype,
            buffer=val,
            strides=val.strides
        )
        if name is not None:
            obj.name = name
        else:
            obj.name = "var_%d" %(Constant.count_unnmaed)
            Constant.count_unnmaed+=1
        
        # gradient related stuff
        obj.grad = np.zeros(obj.shape)

        return obj

class Operation(Node):

    # static dict to keep track of all the operations
    # and unnamed operation nodes
    op_dict = {}

    @staticmethod
    def create_node(
        opres: np.ndarray, opmethod:str,                # meant for result of operation and operation name related
        operand_a, operand_b=None, operand_c = None,    # operands for the operation
        name = None                                     # name of the node
    ):
        '''
        This method creates a variable node and returns it
        Parameters:
        ----------
        opres: number | np.ndarray
            it contains data we want, will be cvt to np.ndarray
            which is the result of the operation

        opmethod: str
            the type of operation we're doing
        
        operand_a: Node
        operand_b: Node | None
        operand_c: Node | None
            these are operands
        
        name: str|None, name of operation
        '''

        obj = Operation(
            shape=opres.shape,
            dtype=opres.dtype,
            buffer=opres,
            strides=opres.strides
        )

        operands = (operand_a, operand_b, operand_c)
        # filter out the None objects
        obj.operands = tuple(operand for operand in operands if operand is not None)
        obj.noperands = len(obj.operands)

        obj.opmethod = opmethod
        
        if name is not None:
            obj.name = name
        
        else:
            if opmethod not in Operation.op_dict:
                Operation.op_dict[opmethod] = 0
            
            node_id = Operation.op_dict[opmethod]
            Operation.op_dict[opmethod]+=1
            obj.name = "%s_%d" %(opmethod, node_id)
        
        # grad related
        obj.grad = np.zeros(obj.shape)

        return obj


# asserting the no.of opernads match is already taken care of when these
# are called
def addbackward(opnode: Operation):
    # opnode = a + b
    if not isinstance(opnode.operands[0],Constant): opnode.operands[0].grad+=opnode.grad * 1
    if not isinstance(opnode.operands[1],Constant): opnode.operands[1].grad+=opnode.grad * 1
def subbackward(opnode: Operation):
    # opnode = a - b
    if not isinstance(opnode.operands[0],Constant): opnode.operands[0].grad+=opnode.grad * 1
    if not isinstance(opnode.operands[1],Constant): opnode.operands[1].grad+=opnode.grad * -1
def mulbackward(opnode: Operation):
    # opnode = a*b
    in0 = opnode.operands[0].view(np.ndarray)
    in1 = opnode.operands[1].view(np.ndarray)
    if not isinstance(opnode.operands[0],Constant): opnode.operands[0].grad+=opnode.grad * in1
    if not isinstance(opnode.operands[1],Constant): opnode.operands[1].grad+=opnode.grad * in0
def divbackward(opnode: Operation):
    # opnode = a / b
    in0 = opnode.operands[0].view(np.ndarray)
    in1 = opnode.operands[1].view(np.ndarray)
    if not isinstance(opnode.operands[0],Constant): opnode.operands[0].grad += opnode.grad*(1/in1)
    if not isinstance(opnode.operands[1],Constant): opnode.operands[1].grad -= opnode.grad*(in0/(in1**2))
def powbackward(opnode: Operation):
    # opnode = a^b
    in0 = opnode.operands[0].view(np.ndarray)
    in1 = opnode.operands[1].view(np.ndarray)
    o = opnode.view(np.ndarray)
    if not isinstance(opnode.operands[0],Constant): opnode.operands[0].grad += opnode.grad*in1*(o/in0)
    if not isinstance(opnode.operands[1],Constant): opnode.operands[1].grad += opnode.grad*np.log(in0)*o
def transposebackward(opnode: Operation):
    if not isinstance(opnode.operands[0], Constant): opnode.operand[0].grad += opnode.grad.T


def backward(a: Node):
    topo=[]
    q = deque()
    map = set()

    a.grad = np.ones(a.shape)
    q.append(a)
    while len(q)>0:
        n = q.popleft()
        ins = n.operands
        map.add(n.name)
        for operand in ins:
            if not isinstance(operand,Operation) or operand.name in map:
                continue
            else:
                q.append(operand)
        topo.append(n)
    
    for node in topo:
        node.grad_fn(node)

def reset():
    Constant.count_unnamed = 0
    Variable.count_unnmaed = 0
    Operation.op_dict.clear() 