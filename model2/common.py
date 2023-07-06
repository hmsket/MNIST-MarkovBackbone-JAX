from jax import numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

    def get_mnist_dataset_of_one_num(self, num):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = train_images[jnp.where((train_labels == num))]
        train_labels = train_labels[jnp.where((train_labels == num))]

        test_images = test_images[jnp.where((test_labels == num))]
        test_labels = test_labels[jnp.where((test_labels == num))]

        train_labels = jnp.where(train_labels==num, 0, 1)
        test_labels = jnp.where(test_labels==num, 0, 1)

        train_labels = jnp.reshape(train_labels, [train_labels.shape[0], -1])
        test_labels = jnp.reshape(test_labels, [test_labels.shape[0], -1])

        # 0以上1以下に正規化
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        return (train_images, train_labels), (test_images, test_labels)

    def get_mnist_dataset_of_two_nums(self, num1, num2):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = train_images[jnp.where((train_labels == num1) | (train_labels == num2))]
        train_labels = train_labels[jnp.where((train_labels == num1) | (train_labels == num2))]

        test_images = test_images[jnp.where((test_labels == num1) | (test_labels == num2))]
        test_labels = test_labels[jnp.where((test_labels == num1) | (test_labels == num2))]

        train_labels = jnp.where(train_labels==num1, 0, 1)
        test_labels = jnp.where(test_labels==num1, 0, 1)

        # 出力層の素子が１個のとき
        # train_labels = jnp.reshape(train_labels, [train_labels.shape[0], -1])
        # test_labels = jnp.reshape(test_labels, [test_labels.shape[0], -1])

        # 出力層の素子が２個以上のとき．（n個とする）
        train_labels = jnp.eye(2)[train_labels]
        test_labels = jnp.eye(2)[test_labels]

        # 0以上1以下に正規化
        train_images = train_images.astype('float32') / 255
        test_images = test_images.astype('float32') / 255
        return (train_images, train_labels), (test_images, test_labels)

    def sigmoid(self, x):
        return 1 / (1+jnp.exp(-x))
    
    def tanh(self, x):
        return (jnp.exp(x)-jnp.exp(-x)) / (jnp.exp(x)+jnp.exp(-x))

    def softmax(self, x, axis):
        max = jnp.max(x, axis=axis, keepdims=True)
        exp_x = jnp.exp(x-max)
        sum_exp_x = jnp.sum(exp_x, axis=axis, keepdims=True)
        new_x = exp_x / sum_exp_x
        return new_x
    
    def plot_area_of_max_weighted_sum(self, conv, params, image):
        conv_w, conv_b = params
        tmp_image = jnp.reshape(image, [1, image.shape[0], image.shape[1]]) # 自作関数を使うための，次元のつじつま合わせ
        hidden = conv.forward(conv_w, conv_b, tmp_image)
        tmp_idxs = jnp.argmax(hidden, axis=2)
        idxs = jnp.reshape(tmp_idxs, tmp_idxs.shape[1]) # ２次元から１次元にする
        
        colors = ['r', 'g', 'b', 'c', 'm'] # 領域を示す線の色を適当に５色用意する

        ax = plt.axes()
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(len(idxs)):
            idx = idxs[i]
            # 畳み込み後の行列の添字に対応する，畳み込み前の行列の添字を計算
            q = idx // (image.shape[0] - conv.kernel_size[0] + 1)
            tmp_idx = idx + q * (conv.kernel_size[0] - 1)
            # ベクトルの添字に対応する，行列にreshapeしたときの座標を計算
            x = tmp_idx // image.shape[0]
            y = tmp_idx % image.shape[0]
            r = patches.Rectangle(xy=(y-0.5,x-0.5), width=conv.kernel_size[0], height=conv.kernel_size[1], fill=False, color=colors[i%5])
            ax.add_patch(r)

        ax.imshow(image, cmap=plt.cm.gray_r)
        plt.show()
    
    def test(self, y, t):
        total = t.shape[0]

        # 出力層の素子が１個のとき
        # y = jnp.where(y < 0, 0, 1)
        
        # 出力層の素子が２個以上のとき
        y = jnp.argmax(y, axis=1)
        t = jnp.argmax(t, axis=1)
        
        ans = jnp.where(y == t, 1, 0)
        acc = jnp.sum(ans) / total
        return acc