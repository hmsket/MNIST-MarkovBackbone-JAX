from jax import numpy as jnp
from tensorflow.keras.datasets import mnist


class F():

    def __init__(self):
        pass

    def get_mnist_dataset(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        train_labels = jnp.eye(10)[train_labels]
        test_labels = jnp.eye(10)[test_labels]
        return (train_images, train_labels), (test_images, test_labels)
    
    def get_mnist_dataset_of_two_nums(self, num1, num2):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = train_images[jnp.where((train_labels == num1) | (train_labels == num2))]
        train_labels = train_labels[jnp.where((train_labels == num1) | (train_labels == num2))]

        test_images = test_images[jnp.where((test_labels == num1) | (test_labels == num2))]
        test_labels = test_labels[jnp.where((test_labels == num1) | (test_labels == num2))]

        train_labels = jnp.where(train_labels==num1, 0, 1)
        test_labels = jnp.where(test_labels==num2, 0, 1)

        # 出力層の素子が１個のとき
        train_labels = jnp.reshape(train_labels, [train_labels.shape[0], -1])
        test_labels = jnp.reshape(test_labels, [test_labels.shape[0], -1])

        # 出力層の素子が２個以上のとき．（n個とする）
        # train_labels = jnp.eye(n)[train_labels]
        # test_labels = jnp.eye(n)[test_labels]

        # 0以上1以下に正規化
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        return (train_images, train_labels), (test_images, test_labels)

    def softmax(self, x, axis):
        max = jnp.max(x, axis=axis, keepdims=True)
        exp_x = jnp.exp(x-max)
        sum_exp_x = jnp.sum(exp_x, axis=axis, keepdims=True)
        new_x = exp_x / sum_exp_x
        return new_x
    
    def test(self, y, t):
        total = t.shape[0]

        # 出力層の素子が１個のとき
        y = jnp.where(y < 0, 0, 1)
        
        # 出力層の素子が２個以上のとき
        # y = jnp.argmax(y, axis=1)
        # t = jnp.argmax(t, axis=1)
        
        ans = jnp.where(y == t, 1, 0)
        acc = jnp.sum(ans) / total
        return acc
