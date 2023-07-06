from jax import random, numpy as jnp


class Conv():

    def __init__(self, nh, nf, stride, kernel_size):
        self.nh = nh # num_hidden
        self.nf = nf # num_filter
        self.stride = stride
        self.kernel_size = kernel_size
    
    def generate_params(self, key):
        w = random.normal(key, shape=[self.nf, self.kernel_size[0]*self.kernel_size[1]])
        b = random.normal(key, shape=[self.nf, 1])
        return w, b
    
    def forward(self, w, b, x):
        col = self.im2col(x)
        tmp = jnp.matmul(w, col)
        y = tmp + b
        return y

    # https://qiita.com/kuroitu/items/35d7b5a4bde470f69570
    def im2col(self, images):
        num_batch = images.shape[0]
        image_size = images.shape[1:] # (28, 28)
        conv_size = ((image_size[0]-self.kernel_size[0]) // self.stride + 1, (image_size[1]-self.kernel_size[1]) // self.stride + 1) # 畳み込みをしたあとの行列のサイズ
        col = jnp.empty([num_batch, self.kernel_size[0], self.kernel_size[1], conv_size[0], conv_size[1]])
        
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                col = col.at[:,i,j].set(images[:,i:i+self.stride*conv_size[0]:self.stride, j:j+self.stride*conv_size[1]:self.stride])
      
        col = jnp.transpose(col, jnp.asarray([1, 2, 0, 3, 4]))
        col = jnp.reshape(col, (self.kernel_size[0]*self.kernel_size[1], num_batch*conv_size[0]*conv_size[1]))
        col = jnp.transpose(col)
        col = jnp.reshape(col, (num_batch, conv_size[0]*conv_size[1], self.kernel_size[0]*self.kernel_size[1]))
        col = jnp.transpose(col, jnp.asarray([0, 2, 1]))
        return col

    def append_off_neuron(self, x):
        num_batch = x.shape[0]
        off_neurons = jnp.zeros([num_batch, 1, self.nh])
        new_x = jnp.hstack([off_neurons, x])
        return new_x

    def get_sum_prob_of_on_neuron(self, x):
        sum = 1 - x[:,0,:]
        return sum
