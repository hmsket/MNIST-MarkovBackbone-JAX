from jax import jit, random, numpy as jnp
from functools import partial


class Linear():

    def __init__(self, ni, no):
        self.ni = ni
        self.no = no

    def generate_params(self, key):
        w = random.normal(key, shape=[self.ni, self.no])
        b = random.normal(key, shape=[self.no])
        return w, b

    @partial(jit, static_argnums=(0,))
    def forward(self, w, b, x):
        tmp = jnp.matmul(x, w)
        y = tmp + b
        return y

    @partial(jit, static_argnums=(0,))
    def append_off_neuron(self, x):
        num_batch = x.shape[0]
        off_neurons = jnp.zeros([num_batch, 1])
        new_x = jnp.hstack([off_neurons, x])
        return new_x
