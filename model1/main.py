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


""" インスタンスを生成 """
conv = Conv(nh, kernel_size)
linear = Linear(nh, no)
F = F()


""" データセットの取得 """
(train_x, train_t), (test_x, test_t) = F.get_mnist_dataset(nums)


""" パラメータの初期生成 """
key, key1 = random.split(random.PRNGKey(seed))
conv_w, conv_b = conv.generate_params(key1)
key, key1 = random.split(key)
linear_w, linear_b = linear.generate_params(key1)
params = [conv_w, conv_b, linear_w, linear_b]


""" 関数の定義 """
def predict(params, x):
    conv_w, conv_b, linear_w, linear_b = params
    tmp = conv.forward(conv_w, conv_b, x)
    tmp = conv.append_off_neuron(tmp)
    tmp = F.softmax(tmp, axis=2)
    tmp = conv.get_sum_prob_of_on_neuron(tmp)
    tmp = linear.forward(linear_w, linear_b, tmp)
    y = F.softmax(tmp, axis=1)
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
loss_fn_jit = jax.jit(loss_fn)
grad_loss = jax.grad(loss_fn, argnums=0)
grad_loss_jit = jax.jit(grad_loss)
update_params_jit = jax.jit(update_params)


""" パラメータの学習 """
max_iter = len(train_t) // n_batch
for i in range(epochs):
    # データセットの順番をシャッフルする
    key, key1 = random.split(key)
    p = random.permutation(key1, len(train_t))
    train_x = train_x[p]
    train_t = train_t[p]

    for iter in range(max_iter):
        batch_x = train_x[iter*n_batch:(iter+1)*n_batch]
        batch_t = train_t[iter*n_batch:(iter+1)*n_batch]
        grads = grad_loss_jit(params, batch_x, batch_t)
        params = update_params_jit(params, grads)
    
    loss = loss_fn_jit(params, train_x, train_t)
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
