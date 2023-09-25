import jax
from jax import random, numpy as jnp

from conv import Conv
from linear import Linear
from common import F



""" ハイパーパラメータの設定 """
nums = [6, 7]        # ６と７の二値分類
nh = 2               # 中間層のブリック数
no = len(nums)       # 出力ブリックの内部状態数
seed = 0             # シード値
mu = 0.8             # 学習係数
kernel_size = [5, 5] # カーネルサイズ
epochs = 30          # エポック数
n_batch = 10         # バッチサイズ
c = 0.001            # おもみをc倍で生成する


""" インスタンスを生成 """
conv = Conv(nh, kernel_size)
linear = Linear(nh, no)
F = F()


""" データセットの取得 """
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset(nums)


""" パラメータの初期生成 """
key, key1 = random.split(random.PRNGKey(seed))
conv_w, conv_b = conv.generate_params(key1, c)
key, key1 = random.split(key)
linear_w, linear_b = linear.generate_params(key1, c)
params = [conv_w, conv_b, linear_w, linear_b]


""" 関数の定義 """
@jax.jit
def predict(params, x, t=1.0):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, t, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    y = F.softmax(tmp, t, axis=1)
    return y

@jax.jit
def loss_fn(params, x, t):    
    y = predict(params, x)
    tmp_loss = -1 * jnp.sum(t*jnp.log(y+1e-7), axis=1) # cross entropy error
    loss = jnp.mean(tmp_loss)
    return loss

@jax.jit
def update_params(params, grads):
    params[0] = params[0] - mu * grads[0]
    params[1] = params[1] - mu * grads[1]
    params[2] = params[2] - mu * grads[2]
    params[3] = params[3] - mu * grads[3]
    return params


""" 関数の事前コンパイル """
grad_loss = jax.grad(loss_fn, argnums=0)
grad_loss_jit = jax.jit(grad_loss)


""" パラメータの学習 """
@jax.jit
def train(idx, params):
    tmp = jax.lax.dynamic_slice_in_dim(p, idx*n_batch, n_batch)    
    grads = grad_loss_jit(params, train_x[tmp], train_t[tmp])
    params = update_params(params, grads)
    return params

max_iter = len(train_t) // n_batch
for i in range(epochs):
    key, key1 = random.split(key)
    p = random.permutation(key1, len(train_t))
    params = jax.lax.fori_loop(0, max_iter, train, params)
    # 現段階での誤差を表示
    loss = loss_fn(params, train_x, train_t)
    print(f'epoch: {i+1} / {epochs}, loss: {loss}')


""" 検証 """
# 学習データでの正解率測定
y = predict(params, train_x)
train_acc = F.test(y, train_t)
print(f'train_acc: {train_acc}')

# 検証データでの正解率測定
y = predict(params, test_x)
test_acc = F.test(y, test_t)
print(f'test_acc : {test_acc}')
