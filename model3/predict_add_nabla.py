import jax
from jax import numpy as jnp

from linear import Linear
from common import F

from tqdm import tqdm
import matplotlib.pyplot as plt


dir = '6_7_tr0.615_te0.621_NI784_NH2_nh5_NO1_no2_s0_m0.8_e1_k5_b1000'
linear1_w = jnp.load(f'./params/{dir}/linear1_w.npy')
linear1_b = jnp.load(f'./params/{dir}/linear1_b.npy')
linear2_w = jnp.load(f'./params/{dir}/linear2_w.npy')
linear2_b = jnp.load(f'./params/{dir}/linear2_b.npy')

hyparams = dir.split('_') # hyper parameters
num1 = int(hyparams[0])
num2 = int(hyparams[1])
NI = int(hyparams[4][2:])
NH = int(hyparams[5][2:])
nh = int(hyparams[6][2:])
NO = int(hyparams[7][2:])
no = int(hyparams[8][2:])
kernel_size = (int(hyparams[12][1:]), int(hyparams[12][1:]))

F = F()
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset_of_two_nums(num1, num2)

linear1 = Linear(NI, NH, nh)
linear2 = Linear(NH, NO, no)


def predict(params, x):
    x = jnp.reshape(x, [-1, NI])
    linear1_w, linear1_b, linear2_w, linear2_b = params
    tmp = linear1.forward(linear1_w, linear1_b, x)
    tmp = linear1.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = linear1.get_sum_prob_of_on_neuron(tmp)
    tmp = jnp.transpose(tmp)
    tmp = linear2.forward(linear2_w, linear2_b, tmp)
    tmp = jnp.reshape(tmp, [-1, no])
    y = F.softmax(tmp, axis=1)
    return y

def predict_add_nabla(params, x, nabla):
    x = jnp.reshape(x, [-1, NI])
    linear1_w, linear1_b, linear2_w, linear2_b = params
    tmp = linear1.forward(linear1_w, linear1_b, x)
    tmp = linear1.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = linear1.get_sum_prob_of_on_neuron(tmp)
    tmp = jnp.transpose(tmp)
    tmp = linear2.forward(linear2_w, linear2_b, tmp)
    tmp = jnp.reshape(tmp, [-1, no])
    y = F.softmax(tmp, axis=1)
    return y

def get_nabla_u_gamma(params, x, gamma):
    x = jnp.reshape(x, [-1, NI])
    linear1_w, linear1_b, linear2_w, linear2_b = params
    tmp = linear1.forward(linear1_w, linear1_b, x)
    tmp = linear1.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = linear1.get_sum_prob_of_on_neuron(tmp)
    tmp = jnp.transpose(tmp)
    tmp = linear2.forward(linear2_w, linear2_b, tmp)
    u_gamma = jnp.reshape(tmp, [-1, no])
    y = u_gamma[0][gamma]
    return y


nabla_fn = jax.grad(get_nabla_u_gamma, argnums=1)

params = [linear1_w, linear1_b, linear2_w, linear2_b]

delta_u_gamma = jnp.empty([no])

for i in range(no):
    nabla = nabla_fn(params, train_x[0:1], i)
    delta_u_gamma = delta_u_gamma.at[i].set(jnp.sum(nabla))

print(delta_u_gamma)
