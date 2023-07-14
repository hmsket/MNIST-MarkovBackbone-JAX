from jax import random, numpy as jnp


class Linear():

    def __init__(self, NI, NO, no):
        self.NI = NI # 入力側のブリック数
        self.NO = NO # 出力側のブリック数
        self.no = no # 内部状態の数，「興奮しない」担当素子を含めない値．

    def generate_params(self, key):
        w = random.normal(key, shape=[self.NO, self.NI, self.no])
        b = random.normal(key, shape=[self.NO, 1, self.no])
        return w, b

    def forward(self, w, b, x):
        tmp = jnp.matmul(x, w)
        y = tmp + b
        return y

    def append_off_neuron(self, x):
        num_batch = x.shape[1]
        off_neurons = jnp.zeros([self.NO, num_batch, 1])
        new_x = jnp.dstack([off_neurons, x])
        return new_x
    
    def get_sum_prob_of_on_neuron(self, x):
        sum = 1 - x[:,:,0]
        return sum
