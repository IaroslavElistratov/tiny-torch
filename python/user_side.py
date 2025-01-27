import ctypes

# import tinytorch as tt
# from tinytorch import Tensor

from tinytorch import *



def test1():
    # now these are python objects
    t1 = Tensor((10, 10))
    t1.print()

    t2 = Tensor((10, 10))
    t2.print()

    t3 = t1 + t2
    t3.print()

    t4 = t3 * t1
    t4.print()

    t4.backward()
    # lib.print()

    t3.print_grad()


# todo: how to specify that it's a pointer to my custom struct?
# conv.argtypes = [POINTER(c_float)]


def test2():
    x = Tensor((2, C, 32, 32))
    kernel1 = Tensor((F, C, HH1, WW1))
    bias_kernel1 = Tensor((F, 1))
    # out = Tensor._from_c_tensor(
    #     batched_conv(x.c_tensor, kernel1.c_tensor, bias_kernel1.c_tensor)
    # )
    # note: wrapped my lib funs into python functions to abstract away the logic in line above
    out = batched_conv(x, kernel1, bias_kernel1)
    out.print()





# test1()
# test2()

# # if you create the instance in a function, it isn't in globals
# kernel1 = Tensor((1, 2, 3, 4))
# print(globals())



# class ConvNet:
#     def __init__(self, B):
#         self.B = B
#         self.C = 3
#         self.F = 6
#         self.F2 = 16
#         self.HH1 = 7
#         self.WW1 = 7

#         self.HH2 = 6
#         self.WW2 = 6

#         self.initialize()


#     def initialize(self):
#         # initialize_conv_net()

#         kernel1 = Tensor((self.F, self.C, self.HH1, self.WW1))
#         kernel1.add_param()
#         bias_kernel1 = Tensor((self.F, 1))
#         bias_kernel1.add_param()

#         kernel2 = Tensor((self.F2, self.F, self.HH2, self.WW2))
#         kernel2.add_param()
#         bias_kernel2 = Tensor((self.F2, 1))
#         bias_kernel2.add_param()

#         w1 = Tensor((self.F2*4*4, 128))
#         w1.add_param()
#         b1 = Tensor((1, 128))
#         b1.add_param()

#         w2 = Tensor((128, 64))
#         w2.add_param()
#         b2 = Tensor((1, 64))
#         b2.add_param()

#         w3 = Tensor((64, 10))
#         w3.add_param()
#         b3 = Tensor((1, 10))
#         b3.add_param()

#         # todo-now: I think param names need to assigned BEFORE calling add_param?
#         kernel1.set_name("kernel1")
#         bias_kernel1.set_name("bias_kernel1")
#         kernel2.set_name("kernel2")
#         bias_kernel2.set_name("bias_kernel2")
#         w1.set_name("w1")
#         b1.set_name("b1")
#         w2.set_name("w2")
#         b2.set_name("b2")
#         w3.set_name("w3")
#         b3.set_name("b3")


#     def forward(self, x):
#         conv1 = batched_conv(x, get_param("kernel1"), get_param("bias_kernel1"))
#         relu1 = relu(conv1)
#         mp1 = batched_maxpool(relu1)

#         conv2 = batched_conv(mp1, get_param("kernel2"), get_param("bias_kernel2"))
#         relu2 = relu(conv2)
#         mp2 = batched_maxpool(relu2)

#         flat = batched_flatten(mp2)

#         # todo-now: implement linear layer in the python abstraction to hide this
#         mm1 = matmul(flat, get_param("w1"))
#         lin1 = add(mm1, repeat(get_param("b1"), axis=0, num_repeats=self.B))
#         relu3 = relu(lin1)

#         mm2 = matmul(relu3, get_param("w2"))
#         lin2 = add(mm2, repeat(get_param("b2"), axis=0, num_repeats=self.B))
#         relu4 = relu(lin2)

#         mm3 = matmul(relu4, get_param("w3"))
#         lin3 = add(mm3, repeat(get_param("b3"), axis=0, num_repeats=self.B))


#         # todo: automate
#         conv1.set_name("conv1")
#         relu1.set_name("relu1")
#         mp1.set_name("mp1")
#         conv2.set_name("conv2")
#         relu2.set_name("relu2")
#         mp2.set_name("mp2")
#         flat.set_name("flat")
#         mm1.set_name("mm1")
#         lin1.set_name("lin1")
#         relu3.set_name("relu3")
#         mm2.set_name("mm2")
#         lin2.set_name("lin2")
#         relu4.set_name("relu4")
#         mm3.set_name("mm3")
#         lin3.set_name("lin3")
#         return lin3




# lib.main()

# B = 2
# C = 3
# x = Tensor((B, C, 32, 32))
# net = ConvNet(B)
# out = net.forward(x)
# # out.print()
# print("functional API: works!\n\n\n")






# comment:
# now using the Linear and Conv layers


class ConvNet(Module):

    # @gc_mark_interval
    def __init__(self):
        # todo: don't initialize params if self.load_params set to True

        # todo: add lazy init to avoid passing "C" for Conv; and "dim_in" for Lin, instead grab a runtime value in forward and initialize weights there
        self.conv1 = Conv(6, 3, 7)
        self.conv2 = Conv(16, 6, 6)

        self.lin1 = Linear(16*4*4, 128)
        self.lin2 = Linear(128, 64)
        self.lin3 = Linear(64, 10)

        # note: the below assumes that GC_IDX was 0 before __init__ ran,
        # so by the end of __init__, GC_IDX in range (0 - self.gc_until) is
        # the internal state of tiny-torch which will not be freed anytime later;
        #   But if GC_IDX idx not 0 at the beginning of __init__, then these tensors will also never gonna be freed

        # todo-high: avoid gc related lines in usr code: add a decorate __init__ (@gc_mark_interval), decorate train_step (@gc_collect)
        # self.gc_until = lib.get_gc_idx()
        # print("net.gc_until: ", self.gc_until)

    # note: if decorate this, won't be call .backward on output of "forward" anymore -- this fn would better be called "inference_step"
    # @gc_collect
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = batched_flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return self.lin3(x, is_activation=False)

    # @gc_collect
    def train_step(self, data):
        x, y = data

        logits = self(x)
        log_probs = log_softmax(logits)
        loss = NLL(log_probs, y)

        loss.backward(zero_grads=True)

        # optimizer step
        lib.adam(LR)

        # lib.accuracy(log_probs.c_tensor, y.c_tensor)

        # lib.free_all_tensors(self.gc_until)
        return loss

    # this should probably be a method on the model, because what params we're saving is tied to this model (these params were created in its __init__)
    @staticmethod
    def save_params():
        lib.save_all_params(LR)

    @staticmethod
    def load_params():
        lib.load_all_params()


# lib.accuracy.argtypes = [ctypes.c_int, ctypes.c_int]
# lib.accuracy.restype = ctypes.c_float


# lib.main()

# lib.print_num_params()
# lib.count_params()

NUM_EPOCHS = 10
BATCH_SIZE = 1024
IS_STOCHASTIC = False
LR = 0.002
# SAVE_EVERY = 1

# todo: make this an argument to get_cifar10
# N_SAMPLES = 1024
# IS_LOAD = False




x = Tensor((16, 3, 32, 32))
net = ConvNet()
out = net.forward(x)
# out.print()
# out.backward()

print("subclass API: works!")

lib.count_params()
lib.print_num_params()

# todo: add argument for the size of the dataset to the C funcs
cifar10_ds = lib.get_cifar10()
x, y = sample_batch(cifar10_ds, BATCH_SIZE, IS_STOCHASTIC)
out = net(x)
# out.print()


for ep_idx in range(NUM_EPOCHS):

    train_batch = sample_batch(cifar10_ds, BATCH_SIZE, IS_STOCHASTIC)

    loss = net.train_step(train_batch)
    loss.print()
    # todo:
    # print(loss.item)

    # if (ep_idx % SAVE_EVERY == 0):
    #     net.save_params()
