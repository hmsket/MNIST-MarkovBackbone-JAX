import jax
from jax import numpy as jnp

from linear import Linear
from common import F

from tqdm import tqdm
import matplotlib.pyplot as plt


"""
自動微分したときの値と，手計算で求めた式による微分した値が一致するかを確認したい
"""

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


# gamma素子の内部状態u_gamma[0][gamma]を，入力train_x[0:1][alpha_y][alpha_x]で偏微分する
gamma = 1
alpha = 389
alpha_x = alpha % 28
alpha_y = alpha // 28


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

def get_nabla_u_gamma(params, x):
    x = jnp.reshape(x, [-1, NI])
    linear1_w, linear1_b, linear2_w, linear2_b = params
    tmp = linear1.forward(linear1_w, linear1_b, x)
    tmp = linear1.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = linear1.get_sum_prob_of_on_neuron(tmp)
    tmp = jnp.transpose(tmp)
    tmp = linear2.forward(linear2_w, linear2_b, tmp)
    u_gamma = jnp.reshape(tmp, [-1, no])
    # y = u_gamma[0][0]
    y = u_gamma[0][gamma]
    return y

def get_x_beta(params, x):
    x = jnp.reshape(x, [-1, NI])
    linear1_w, linear1_b, linear2_w, linear2_b = params
    tmp = linear1.forward(linear1_w, linear1_b, x)
    tmp = linear1.append_off_neuron(tmp)
    x_beta = F.softmax(tmp, axis=2)
    return x_beta


params = [linear1_w, 0, linear2_w, 0] # bias=0にしておく．手計算のときにbiasを考慮していないから．
# params = [linear1_w, linear1_b, linear2_w, linear2_b]


""" u_gamma_0を，入力train_x[0]で偏微分したときの値 """
nabla_fn = jax.grad(get_nabla_u_gamma, argnums=1)
nabla = nabla_fn(params, train_x[0:1]) # shape: (batch_size, 28, 28)
# print(f'自動微分: {nabla[0][0][0]}')
print(f'自動微分: {nabla[0][alpha_y][alpha_x]}') # 入力画像の(alpha_y, alpha_x)で偏微分した値


""" (18)式どおりに計算した値 """
x_beta = get_x_beta(params, train_x[0:1])

sum = 0
for l in range(NH):
    for i in range(nh):
        for j in range(nh):
            if i == j:
                # x_beta[l][0][0]は興奮しない素子というデータ構造なため，x_beta[l][0][j+1]と+1することで辻褄が合う．
                # sum = sum + linear2_w[0][l][0]*(x_beta[l][0][j+1]-x_beta[l][0][i+1]*x_beta[l][0][j+1])*linear1_w[l][0][j]
                sum = sum + linear2_w[0][l][gamma]*(x_beta[l][0][j+1]-x_beta[l][0][i+1]*x_beta[l][0][j+1])*linear1_w[l][alpha][j]
            else:
                # sum = sum + linear2_w[0][l][0]*(-1*x_beta[l][0][i+1]*x_beta[l][0][j+1])*linear1_w[l][0][j]
                sum = sum + linear2_w[0][l][gamma]*(-1*x_beta[l][0][i+1]*x_beta[l][0][j+1])*linear1_w[l][alpha][j]
print(f'式(18)  : {sum}')
