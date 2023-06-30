# MNIST-MarkovBackbone-JAX
多値を取る素子を用いた神経回路モデルで，MNISTの認識をする．

## model
![二値分類をしたいときのモデル例](https://github.com/hmsket/MNIST-MarkovBackbone-JAX/assets/74644437/42e7e096-6475-4a78-95b1-6d235420a8f2)

## code
### main.py
学習をおこなったあと，パラメータを保存する．ただそれだけ．なにか図を作るわけではない．

### anime.py
学習をおこなう際，畳み込みフィルタの移り変わりアニメーションを作成する．パラメータも保存する．  
[注意]：中間層の素子$z^{\beta_k}$の数$N_h$によって，アニメーションのタイトルの位置が変わる．職人的な位置調整が必要．コードをたらたらと書けば，人力じゃなくて自動で調整できるんだけど(^o^)いつかやろう(^o^)  
![w_conv_anime](https://github.com/hmsket/MNIST-MarkovBackbone-JAX/assets/74644437/1bc8fbf5-0edf-44ac-b00f-cff448208899)
