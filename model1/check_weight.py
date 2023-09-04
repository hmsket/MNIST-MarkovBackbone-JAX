from jax import numpy as jnp

from conv import Conv
from linear import Linear



# 事前に学習したパラメータファイルを読み込む
dir = '6,7_tr0.968_te0.972_nh2_no2_s0_m0.8_e30_k5_b10'

hyparams = dir.split('_') # hyper parameters
nums = list(map(int, hyparams[0].split(',')))
nh = int(hyparams[3][2:])
no = int(hyparams[4][2:])
kernel_size = (int(hyparams[8][1:]), int(hyparams[8][1:]))

conv = Conv(nh, kernel_size)
linear = Linear(nh, no)

conv_w = jnp.load(f'./params/{dir}/conv_w.npy')
conv_b = jnp.load(f'./params/{dir}/conv_b.npy')
linear_w = jnp.load(f'./params/{dir}/linear_w.npy')
linear_b = jnp.load(f'./params/{dir}/linear_b.npy')


for i in range(nh):
    print(f'w_beta_{i+1}:')
    for j in range(kernel_size[0]):
        for k in range(kernel_size[1]):
            w = conv_w[i][kernel_size[0]*j+k]
            if w < 0:
                print(f'{w:.2f}, ', end='')
            else:
                print(f' {w:.2f}, ', end='')
        print()
    print()
