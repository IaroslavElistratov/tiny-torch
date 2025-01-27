import os
import ctypes

device = os.environ.get('DEVICE', None)
if device == "CPU":
    # OSError: ./libtinytorch.so: undefined symbol: _ZNSt8ios_base4InitD1Ev
    # This is a problem that usually occurs when loading in a shared library and then trying to code from that library. 
    # Since linking of the library did not happen directly after compile time, you will not get any linker errors that would be associated with the file and what is dynamically loading it.
    # Look into a demangler(c++filt) and try to get more information about what that "_ZNSt8ios_base4InitD1Ev" references.
    lib = ctypes.CDLL("./libtinytorch_cpu.so")
    # from ctypes.util import find_library
    # lib = ctypes.CDLL(find_library("tinytorch"))
elif device == "CUDA":
    lib = ctypes.CDLL("./libtinytorch_cuda.so")
else:
    raise ValueError("Expected DEVICE variable ('CUDA' or 'CPU') to be set")


# # native:
# t1 = lib.Tensor2d(10, 10)
# lib.print(t1)
# t2 = lib.Tensor2d(10, 10)
# lib.print(t2)

# t3 = lib.add_k(t1, t2)
# lib.print(t3)




# args = (10, 10)
# # lib.Tensor2d(*args)

# def create_tensor(args):
#     return lib.Tensor2d(*args)

# create_tensor(args)



# re-implement macros:
def lib_Tensor(args):
    if len(args) == 2:
        return lib.Tensor2d(*args)
    elif len(args) == 3:
        return lib.Tensor3d(*args)
    elif len(args) == 4:
        return lib.Tensor4d(*args)
    else:
        print("lib.Tensor: unsupported num args")


def lib_TensorNoData(args):
    if len(args) == 2:
        return lib.TensorNoData2d(*args)
    elif len(args) == 3:
        return lib.TensorNoData3d(*args)
    elif len(args) == 4:
        return lib.TensorNoData4d(*args)
    else:
        print("lib.TensorNoData: unsupported num args")


def lib_EmptyTensor(args):
    if len(args) == 2:
        return lib.EmptyTensor2d(*args)
    elif len(args) == 3:
        return lib.EmptyTensor3d(*args)
    elif len(args) == 4:
        return lib.EmptyTensor4d(*args)
    else:
        print("lib.EmptyTensor: unsupported num args")


# def lib_slice(args):
#     if len(args) == 3:
#         return lib.slice_2d(*args)
#     elif len(args) == 4:
#         return lib.slice_3d(*args)
#     elif len(args) == 5:
#         return lib.slice_4d(*args)
#     else:
#         print("lib.slice: unsupported num args")


# def lib_view(args):
#     if len(args) == 3:
#         return lib.view_2d(*args)
#     elif len(args) == 4:
#         return lib.view_3d(*args)
#     elif len(args) == 5:
#         return lib.view_4d(*args)
#     else:
#         print("lib.view: unsupported num args")



# # https://github.com/tensorflow/tensorflow/blob/c30bef1d00717bc5de52b994d8a2d7a2e590fef5/tensorflow/python/ops/ragged/ragged_operators.py#L252
# def _right(operator):
#   # Right-handed version of an operator: swap args x and y.
#   return lambda y, x: operator(x, y)






class Tensor:
    # def __init__(self, args=None, c_tensor=None):
    #     # needed "c_tensor" to bypass creating a new C tensor, which is needed when a new C tensor is produced as a result of
    #     # calling come C function (e.g. lib.add_k). At which point the C tensor has already been created and we just
    #     # need to wrap that existing c tensor into a python class
    #     assert (args is not None) or (c_tensor is not None)
    #     if c_tensor:
    #         self.c_tensor = c_tensor
    #     else:
    #         self.c_tensor = lib.Tensor2d(*args)

    # the new __init__ + from_c_tensor methods, as oppose to the __init__ above this allows to only have a
    # single argument in __init__, hiding the internal logic for initializing from_c_tensor into a separate class
    def __init__(self, args):
        self.c_tensor = lib_Tensor(args) # lib.Tensor2d(*args)
        # # todo-now: setname with python instance name
        # #    - overwrite assignment operator?
        # lib.set_name(self.c_tensor, create_string_buffer(b"abc"))
        # # https://stackoverflow.com/a/32163468

        # todo: variable names exist in the namespace where they are defined, not within the objects they reference
        # print(globals())
        # d = {v:k for k,v in globals().items()}
        # print("instance name: ", d[self])
        # return d[self]

    def set_name(self, name):
        # lib.set_name(self.c_tensor, ctypes.create_string_buffer(b"kernel1"))
        b = bytes(name, encoding="utf-8")
        lib.set_name(self.c_tensor, ctypes.create_string_buffer(b))

    def add_param(self):
        lib.add_param(self.c_tensor)

    @staticmethod
    def _from_c_tensor(c_tensor):
        # circumvent passing c_tensor to __init__
        py_tensor = Tensor.__new__(Tensor)
        py_tensor.c_tensor = c_tensor
        return py_tensor

    def print(self):
        # Note that printf prints to the real standard output channel, not to sys.stdout, so these examples will only work at the console prompt, not from within IDLE:
        lib.print(self.c_tensor)

    def print_grad(self):
        print("grads:")
        lib.print_grad(self.c_tensor)

    def backward(self, zero_grads):
        if zero_grads:
            lib.zero_grads()
        lib.save_num_uses(self.c_tensor)
        lib.backward(self.c_tensor)

    # @property
    # def size(self):
    #     return lib.get_size(self.c_tensor)

    @property
    def rank(self):
        return lib.get_rank(self.c_tensor)

    @property
    def shape(self):
        shape = list()
        # "-1" bc e.g. a tensor of size 3, will have dims in range 0-2
        for dim_idx in range(self.rank):
            shape.append(
                lib.get_shape_at_idx(self.c_tensor, dim_idx)
            )
        return tuple(shape)

    @property
    def item(self):
        return lib.item(self.c_tensor)

    # todo: check if other is a tensor
    def __add__(self, other):
        # return Tensor(c_tensor=lib.add_k(self.c_tensor, other.c_tensor))
        return Tensor._from_c_tensor(lib.add(self, other))

    # todo: automatically reverse __add__
    # def __radd__(self, other):
    #     pass

    # def __getitem__(self, other):
    #     return index(self)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    # todo: is it this in my case? Or __div__?
    def __truediv__(self, other):
        return div(self, other)

    def __pow__(self, power):
        return pow(self, power)

    def __neg__(self):
        return neg(self)

# lib.item.restype = ctypes.c_float
lib.TensorLikeFill.argtypes = [ctypes.c_int, ctypes.c_float]

class TensorLikeFill(Tensor):
    def __init__(self, tensor, fill_value):
        self.c_tensor = lib.TensorLikeFill(tensor.c_tensor, fill_value)

    # these are constructors so no need for "_from_c_tensor" method
    def _from_c_tensor(self, c_tensor):
        raise ValueError("TensorLikeFill is tensor constructor -- it does not support ._from_c_tensor")


lib.TensorScalarFill.argtypes = [ctypes.c_float]

class TensorScalarFill(Tensor):
    def __init__(self, fill_value):
        self.c_tensor = lib.TensorScalarFill(fill_value)

    def _from_c_tensor(self, c_tensor):
        raise ValueError("TensorLikeFill is tensor constructor -- it does not support ._from_c_tensor")





# todo-now:
# wrap the below functions in a couple line python fns, to provide
#   - wrap output into py tensor: Tensor._from_c_tensor()
#   - use t.c_tensor to pass as inputs to the C fns
#   - kwarg support

# tiny torch has these python-overloadable ops:

# add = lib.add
# sub = lib.sub
# mul = lib.mul
# div = lib.div
# neg = lib.neg
# pow = lib.pow

# todo-now: implement unary and binary op decorators: abstract much of the below into the decorator
def add(t1, t2):
    return Tensor._from_c_tensor(lib.add(t1.c_tensor, t2.c_tensor))

def add_k_(t1, t2, t3):
    return Tensor._from_c_tensor(lib.add_k_(t1.c_tensor, t2.c_tensor, t3.c_tensor))


def sub(t1, t2):
    return Tensor._from_c_tensor(lib.sub(t1.c_tensor, t2.c_tensor))

def mul(t1, t2):
    return Tensor._from_c_tensor(lib.mul(t1.c_tensor, t2.c_tensor))

def div(t1, t2):
    return Tensor._from_c_tensor(lib.div(t1.c_tensor, t2.c_tensor))

def neg(t1):
    return Tensor._from_c_tensor(lib.neg(t1.c_tensor))

def pow(t1, power):
    return Tensor._from_c_tensor(lib.pow(t1.c_tensor, power))



# cannot overload these:

# log = lib.log
# exp = lib.exp
# repeat = lib.repeat
# select = lib.select
# relu = lib.relu
# transpose = lib.transpose
# batched_flatten = lib.batched_flatten

# matmul = lib.matmul
# batched_matmul = lib.batched_matmul

# reduce_sum = lib.reduce_sum
# batched_reduce_sum = lib.batched_reduce_sum

# reduce_max = lib.reduce_max
# batched_reduce_max = lib.batched_reduce_max

# conv = lib.conv
# batched_conv = lib.batched_conv

# maxpool = lib.maxpool
# batched_maxpool = lib.batched_maxpool


def log(t1):
    return Tensor._from_c_tensor(lib.log(t1.c_tensor))


def exp(t1):
    return Tensor._from_c_tensor(lib.exp(t1.c_tensor))


def relu(t1):
    return Tensor._from_c_tensor(lib.relu(t1.c_tensor))

def transpose(t1):
    return Tensor._from_c_tensor(lib.transpose(t1.c_tensor))


def repeat(t1, axis, num_repeats):
    return Tensor._from_c_tensor(lib.repeat(t1.c_tensor, axis, num_repeats))



# todo: expose t1.n_dims as python int -- this will allow to hide batched_ in the python level: change this fn to dispatch to either conv or batched_conv, based on t->num_dims
def matmul(input, kernel):
    return Tensor._from_c_tensor(lib.matmul(input.c_tensor, kernel.c_tensor))

def batched_matmul(input, kernel):
    return Tensor._from_c_tensor(lib.batched_matmul(input.c_tensor, kernel.c_tensor))


def conv(input, kernel, bias):
    return Tensor._from_c_tensor(lib.conv(input.c_tensor, kernel.c_tensor, bias.c_tensor))

def batched_conv(input, kernel, bias):
    return Tensor._from_c_tensor(lib.batched_conv(input.c_tensor, kernel.c_tensor, bias.c_tensor))



def maxpool(t1):
    return Tensor._from_c_tensor(lib.maxpool(t1.c_tensor))

def batched_maxpool(t1):
    return Tensor._from_c_tensor(lib.batched_maxpool(t1.c_tensor))



def reduce_sum(t1):
    return Tensor._from_c_tensor(lib.reduce_sum(t1.c_tensor))

def batched_reduce_sum(t1):
    return Tensor._from_c_tensor(lib.batched_reduce_sum(t1.c_tensor))


def reduce_max(t1):
    return Tensor._from_c_tensor(lib.reduce_max(t1.c_tensor))

def batched_reduce_max(t1):
    return Tensor._from_c_tensor(lib.batched_reduce_max(t1.c_tensor))


lib.select.argtypes = [ctypes.c_int, ctypes.c_int]

def select(input, idx):
    return Tensor._from_c_tensor(lib.select(input.c_tensor, idx.c_tensor))



def batched_flatten(t1):
    return Tensor._from_c_tensor(lib.batched_flatten(t1.c_tensor))



### Composite ops

# def log_softmax(t1):
#     return Tensor._from_c_tensor(lib.log_softmax(t1.c_tensor))

# def NLL(t1, t2):
#     return Tensor._from_c_tensor(lib.NLL(t1.c_tensor, t2.c_tensor))


def log_softmax(logits):
    # min-max trick for numerical stability, python: "logits -= np.max(logits, axis=1, keepdims=True)"
    n_repeats = logits.shape[1]
    maxes = repeat(batched_reduce_max(logits), axis=1, num_repeats=n_repeats)
    su = sub(logits, maxes)

    ex = exp(su)                           # (B, ?)
    re = batched_reduce_sum(ex)            # (B, 1)

    add_k_(re, TensorLikeFill(re, 6e-8), re)
    denom = log(re)                        # (B, 1)

    n_repeats = ex.shape[1]
    denom_broadcasted = repeat(denom, axis=1, num_repeats=n_repeats)

    log_sm = sub(su, denom_broadcasted)    # (B, ?)

    # set_name(maxes, "maxes")
    # set_name(su, "su")
    # set_name(ex, "ex")
    # set_name(re, "re")
    # set_name(denom, "denom")
    # set_name(denom_broadcasted, "denom_broadcasted")
    # set_name(log_sm, "log_sm")
    return log_sm


def NLL(log_probs, label):
    B = label.shape[0]
    se = select(log_probs, label)                   # (B, 1)
    lgsum = reduce_sum(se)                          # (, )
    nll = neg(lgsum)                                # (, )
    # divide by the batch size
    nll_normalized = div(nll, TensorScalarFill(B))  # (, )

    # set_name(label, "label")
    # set_name(se, "se")
    # set_name(lgsum, "lgsum")
    # set_name(nll, "nll")
    # set_name(nll_normalized, "nll_normalized")
    return nll_normalized










# impl'ed c function to extract Tensor* from Cifar10* -- like so, cifar10_batch->input (bc can't do this in python)
# and same for y=batch->label
def _unpack_cifar(cifar10_batch):
    x = Tensor._from_c_tensor(lib.unpack_cifar_x(cifar10_batch))
    y = Tensor._from_c_tensor(lib.unpack_cifar_y(cifar10_batch))
    return x, y

def sample_batch(cifar10_ds, batch_size, is_random):
    cifar10_batch = lib.sample_batch(cifar10_ds, batch_size, is_random)
    return _unpack_cifar(cifar10_batch)



# lib.Tensor2d.argtypes = [ctypes.c_int, ctypes.c_int]



# # note:
# # all Python types except integers, strings, and bytes objects have to be wrapped in their corresponding ctypes type, so that they can be converted to the required C data type:


# # todo-now:
# # By default functions are assumed to return the C int type. Other return types can be specified by setting the restype attribute of the function object.

# # lib.select_one(4, 10)

# t5 = Tensor((10, 10, 10, 10))
# t5.print()


# initialize_conv_net = lib.initialize



# wt this would segfault if provide wrong argument type (e.g. python string)
lib.Tensor2d.argtypes = [ctypes.c_int, ctypes.c_int]

# question-now: how does ctypes.c_int automatically work to represent my "Tensor*"
# used in TensorConstructor
lib.set_name.argtypes = [ctypes.c_int, ctypes.c_char_p]



lib.get_param.argtypes = [ctypes.c_char_p]

def get_param(name):
    b = bytes(name, encoding="utf-8")
    return Tensor._from_c_tensor(lib.get_param(ctypes.create_string_buffer(b)))




COUNTER_LINEAR = 0
COUNTER_CONV = 0

# todo: ugly
# makes param names unique -- make get_param (in forward) use that unique param name
def increment_counter(self):
    # pre increment count
    count = None
    if type(self).__name__ == "Linear":
        global COUNTER_LINEAR
        count = COUNTER_LINEAR
        COUNTER_LINEAR += 1
    elif type(self).__name__ == "Conv":
        global COUNTER_CONV
        count = COUNTER_CONV
        COUNTER_CONV += 1
    else:
        raise ValueError("Expected only Linear or Conv")
    print(f"incremented {type(self).__name__} counter")
    # self.counter = count
    return count


# todo: automatically call .initialize on all the children -- currently .initialize is too verbose;
# Create a Module class and add that recursive traversal method to that class,
# then make all my layers subclass that Module class to inherit this functionality
class Module:
    def __init__():
        pass

    # def initialize(self):
    #     self.counter = increment_counter(self)

    # todo: add pre/post hooks
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.initialize()

    def initialize(self):
        self.counter = increment_counter(self)

        w = Tensor((self.dim_in, self.dim_out))
        w.set_name(f"weight_{self.counter}")
        w.add_param()

        b = Tensor((1, self.dim_out))
        b.set_name(f"bias_{self.counter}")
        b.add_param()

    def forward(self, x, is_activation=True):
        B = x.shape[0]
        x = matmul(x, get_param(f"weight_{self.counter}"))
        # repeat below does explicit broadcasting
        x = add(x, repeat(get_param(f"bias_{self.counter}"), axis=0, num_repeats=B))
        if is_activation: return relu(x)
        else: return x



class Conv(Module):
    def __init__(self, F, C, K):
        self.F = F
        self.C = C
        self.K = K

        self.initialize()

    def initialize(self):
        self.counter = increment_counter(self)

        kernel = Tensor((self.F, self.C, self.K, self.K))
        kernel.set_name(f"kernel_{self.counter}")
        kernel.add_param()

        bias = Tensor((self.F, 1))
        # note: C t->name must be unique as to not conflict with biases of the Linear layers
        bias.set_name(f"bias_kernel_{self.counter}")
        bias.add_param()

    def forward(self, x):
        x = batched_conv(
            x,
            get_param(f"kernel_{self.counter}"),
            get_param(f"bias_kernel_{self.counter}")
        )
        x = relu(x)
        x = batched_maxpool(x)
        return x


lib.adam.argtypes = [ctypes.c_float]
