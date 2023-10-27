import numpy as np
from jax import numpy as jnp

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



# MNISTのうち，配列labelsで指定したラベルのみのデータセットを作成する関数
def get_mnist_dataset(labels):
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_idx = jnp.zeros(len(train_y))
    test_idx = jnp.zeros(len(test_y))

    for label in labels:
        train_idx = train_idx + (train_y == label)
        test_idx = test_idx + (test_y == label)

    train_x = train_x[jnp.where(train_idx)]
    train_y = train_y[jnp.where(train_idx)]

    test_x = test_x[jnp.where(test_idx)]
    test_y = test_y[jnp.where(test_idx)]

    # ラベルの値を変える
    # e.g. labels=[6,7]のとき，6を0,7を1に変える
    # こうしないと，to_categorical()が使えない
    for i, label in enumerate(labels):
        np.place(train_y, (train_y==label)>0, i)
        np.place(test_y, (test_y==label)>0, i)
 
    # 画像のピクセル値を，0以上1以下に正規化
    train_x = train_x.astype('float32') / 255
    test_x = test_x.astype('float32') / 255

    # 望ましい値を，one-hotベクトルにする
    N = len(labels)
    train_y = to_categorical(train_y, N) # ２番目の引数で，次元数を指定する
    test_y = to_categorical(test_y, N)

    # numpyの配列を，jax.numpyの配列に型変換する
    train_x = jnp.array(train_x)
    train_y = jnp.array(train_y)
    test_x = jnp.array(test_x)
    test_y = jnp.array(test_y)

    return (train_x, train_y), (test_x, test_y)


def softmax(x, axis, t=1.0):
    max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp((x-max)/t)
    sum_exp_x = jnp.sum(exp_x, axis=axis, keepdims=True)
    new_x = jnp.divide(exp_x, sum_exp_x)
    return new_x


def test(z, y):
    N = len(y)
    z = jnp.argmax(z, axis=1)
    y = jnp.argmax(y, axis=1)
    ans = jnp.where(z==y, 1, 0)
    acc = jnp.sum(ans) / N
    return acc
