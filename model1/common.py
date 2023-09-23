from jax import numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.keras.datasets import mnist



class F():

    def __init__(self):
        pass
    
    # MNISTデータセットのうち，リスト変数num_listで与えられたラベルのみ返す
    def get_mnist_dataset(self, num_list):
        (train_images_tmp, train_labels_tmp), (test_images_tmp, test_labels_tmp) = mnist.load_data()

        train_idx = jnp.zeros(len(train_labels_tmp))
        test_idx = jnp.zeros(len(test_labels_tmp))

        for num in num_list:
            train_idx = train_idx + (train_labels_tmp == num)
            test_idx = test_idx + (test_labels_tmp == num)

        train_images_tmp = train_images_tmp[jnp.where(train_idx)]
        train_labels_tmp = train_labels_tmp[jnp.where(train_idx)]

        test_images_tmp = test_images_tmp[jnp.where(test_idx)]
        test_labels_tmp = test_labels_tmp[jnp.where(test_idx)]

        # 画像のピクセル値を，0以上1以下に正規化
        train_images = train_images_tmp.astype('float32') / 255
        test_images = test_images_tmp.astype('float32') / 255

        # ラベルのone-hotベクトル化
        train_labels = jnp.zeros(len(train_labels_tmp))
        test_labels = jnp.zeros(len(test_labels_tmp))
        for i in range(len(num_list)):
            train_labels = jnp.where(train_labels_tmp==num_list[i], i, train_labels)
            test_labels = jnp.where(test_labels_tmp==num_list[i], i, test_labels)
        train_labels = jnp.eye(len(num_list))[jnp.int8(train_labels)]
        test_labels = jnp.eye(len(num_list))[jnp.int8(test_labels)]

        train_images = jnp.array(train_images)
        train_labels = jnp.array(train_labels)
        test_images = jnp.array(test_images)
        test_labels = jnp.array(test_labels)

        return (train_images, train_labels), (test_images, test_labels)

    def softmax(self, x, t, axis):
        max = jnp.max(x, axis=axis, keepdims=True)
        exp_x = jnp.exp((x-max)/t)
        sum_exp_x = jnp.sum(exp_x, axis=axis, keepdims=True)
        new_x = jnp.divide(exp_x, sum_exp_x)
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
