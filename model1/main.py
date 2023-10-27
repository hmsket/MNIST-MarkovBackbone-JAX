import jax
from jax import random, numpy as jnp

from conv import Conv
from linear import Linear
from common import *



""" ハイパーパラメータの設定 """
labels = [6, 7]
nh = 2               # 中間層のブリック数
no = len(labels)     # 出力ブリックの内部状態数
seed = 0             # シード値
mu = 0.8             # 学習係数
kernel_size = [5, 5] # カーネルサイズ
epochs = 30          # エポック数
n_batch = 10         # バッチサイズ


""" インスタンスを生成 """
conv = Conv(nh, kernel_size)
linear = Linear(nh, no)


""" データセットの取得 """
(train_x, train_y), (test_x, test_y) = get_mnist_dataset(labels)


""" パラメータの初期生成 """
key, key1 = random.split(random.PRNGKey(seed))
conv_w, conv_b = conv.generate_params(key1)
key, key1 = random.split(key)
linear_w, linear_b = linear.generate_params(key1)
params = [conv_w, conv_b, linear_w, linear_b]


""" 関数の定義 """
@jax.jit
def predict(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = softmax(tmp, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    z = softmax(tmp, axis=1)
    return z

@jax.jit
def loss_fn(params, x, y):
    z = predict(params, x)
    tmp_loss = -1 * jnp.sum(y*jnp.log(z+1e-7), axis=1) # cross entropy error
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
max_iter = len(train_y) // n_batch
for i in range(epochs):
    # データセットの順番をシャッフルする
    key, key1 = random.split(key)
    p = random.permutation(key1, len(train_y))
    train_x = train_x[p]
    train_y = train_y[p]

    for iter in range(max_iter):
        batch_x = train_x[iter*n_batch:(iter+1)*n_batch]
        batch_y = train_y[iter*n_batch:(iter+1)*n_batch]
        grads = grad_loss_jit(params, batch_x, batch_y)
        params = update_params(params, grads)
    
    loss = loss_fn(params, train_x, train_y)
    print(f'epoch: {i+1} / {epochs}, loss: {loss}')


""" 検証 """
# 学習データでの正解率測定
z = predict(params, train_x)
train_acc = test(z, train_y)
print(f'train_acc: {train_acc}')

# 検証データでの正解率測定
z = predict(params, test_x)
test_acc = test(z, test_y)
print(f'test_acc : {test_acc}')
