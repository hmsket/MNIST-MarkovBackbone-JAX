# MNIST-MarkovBackbone-JAX
多値を取る素子を用いた神経回路モデルで，MNISTの認識をする．

## model1
中間層の素子ごとにフィルタを用意し，入力画像を走査しながら畳み込む．
![model1](https://github.com/hmsket/MNIST-MarkovBackbone-JAX/assets/74644437/42e7e096-6475-4a78-95b1-6d235420a8f2)

## model2
中間層の素子ごとに担当する入力画像の領域を変え，あらかじめ用意された８種類のフィルタで畳み込む．
![model2](https://github.com/hmsket/MNIST-MarkovBackbone-JAX/assets/74644437/afc2f6c5-6ff4-42bd-bcd8-6e6f3f1ddf56)
