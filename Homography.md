# 画像の写像

画像の写像についてメモ。後々もっときちんと知識を整理したい。

実践コンピュータビジョン  
https://www.oreilly.co.jp/books/9784873116075/

3章

Multiple View Geometry in Computer Vision
Second Edition  
https://www.robots.ox.ac.uk/~vgg/hzbook/

2-4章

## ホモグラフィー

ホモグラフィーはある平面から別の平面へ2次元上の射影変換のことである。このホモグラフィーを理解するに当たって、線形変換、平行移動と同次座標系の導入、アフィン変換、射影変換と理解していくと分かりやすい。

### 線形変換

[Python, OpenCVで幾何変換 - note.nkmk.me](https://note.nkmk.me/python-opencv-warp-affine-perspective/)

に詳しい解説がある。


2次元上の座標$(x,y)$を$(x',y')$に移動するとき、次の式で表される変換を線形変換という。
$$
\begin{pmatrix}
x'\\
y'
\end{pmatrix}
=\begin{pmatrix}
a & b\\
c & d
\end{pmatrix}
\begin{pmatrix}
x\\
y
\end{pmatrix}
$$
画像上の座標を線形変換することで、拡大・縮小、回転、スキュー(せん断)ができる。

### 同時座標で表す変換

一方、座標を平行移動しようとすると、次のようになる。
$$
\begin{aligned}
x'=x+t_x\\
y'=y+t_y\\
\end{aligned}
$$
これは線形変換では表せない。平行移動を表現するために、座標$(x,y)$に対して1つ要素を付け加え、$(x,y,\omega)$とした同次座標系(斉次座標系)を導入する。  
同次座標系は定数倍しても同じ2次元上の座標を表す。つまり$(x,y,\omega)$も$(\alpha x,\alpha y,\alpha\omega)$も$(x/\omega,y/\omega,1)$も同じ2次元上の座標を表す。   
この同次座標系を使えば、平行移動は次のようになる。
$$
\begin{pmatrix}
x'\\
y'\\
1
\end{pmatrix}
=\begin{pmatrix}
1 & 0 & t_x\\
0 & 1 & t_y\\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x\\
y\\
1
\end{pmatrix}
$$
線形変換は次のようになる。
$$
\begin{pmatrix}
x'\\
y'\\
1
\end{pmatrix}
=\begin{pmatrix}
a & b & 0\\
c & d & 0\\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x\\
y\\
1
\end{pmatrix}
$$

### アフィン変換

線形変換と平行移動を組み合わせた変換はアフィン変換と呼ばれ次のようになる。
$$
\begin{pmatrix}
x'\\
y'\\
1
\end{pmatrix}
=\begin{pmatrix}
a & b & t_x\\
c & d & t_y\\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x\\
y\\
1
\end{pmatrix}
$$

アフィン変換によって画像を拡大・縮小、回転、スキューと平行移動が一度に計算できることになった。

### 射影変換

アフィン変換は行列の$2\times3$の部分しか使っていないが、$3\times3$全てを使う変換を射影変換と呼び、次のようになる。
$$
\begin{pmatrix}
x'\\
y'\\
\omega'
\end{pmatrix}
=\begin{pmatrix}
h_1 & h_2 & h_3\\
h_4 & h_5 & h_6\\
h_7 & h_8 & h_9
\end{pmatrix}
\begin{pmatrix}
x\\
y\\
\omega
\end{pmatrix}
$$
射影変換はアフィン変換よりいろいろな変換を表わせ、例えば長方形から台形への変換なども表せる。この射影変換は、同次座標$\pmb{x}$とホモグラフィー行列$H$とを使って次のように表すこととする。
$$
\pmb{x}'=H\pmb{x}
$$

## DLTアルゴリズム

行いたい変換から直接ホモグラフィー行列$H$を作るのは難しいので、変換前の座標と変換後の座標からホモグラフィー行列を計算できたら便利。  
DLT(Direct Linear Transformation:直接線形変換)法は、4つ以上の点の対応を使って$H$を計算するアルゴリズムである。対応する座標同士を$(x_1,y_1)$と$(x_1',y_1')$、$(x_2,y_2)$と$(x_2',y_2')$、$\cdots$とすると$H$を使って対応点を変換する式は次のように書き換えられる。
$$
\begin{pmatrix}
-x_1 & -y_1 & -1 & 0 & 0 & 0 & x_1x_1' & y_1x_1' & x_1'\\
0 & 0 & 0 & -x_1 & -y_1 & -1 & x_1y_1' & y_1y_1' & y_1'\\
-x_2 & -y_2 & -1 & 0 & 0 & 0 & x_2x_2' & y_2x_2' & x_2'\\
0 & 0 & 0 & -x_1 & -y_1 & -1 & x_2y_2' & y_2y_2' & y_2'\\
&\vdots&&\vdots&&\vdots&&\vdots
\end{pmatrix}
\begin{pmatrix}
h_1 \\ h_2 \\ h_3\\
h_4 \\ h_5 \\ h_6\\
h_7 \\ h_8 \\ h_9
\end{pmatrix}
=\pmb{0}
$$
と$Ah=0$という形になる。この式について特異値分解(Singular Value Decomposition: SVD)を用いて$H$を求めることができる。

**導出**

ホモグラフィー行列$H$の要素を次のように表す。($h^{1T}$の$1$は添字で$T$は転置の意味)
$$
\begin{aligned}
\pmb{h}^{1T}=\left(h_1,h_2,h_3\right)\\
\pmb{h}^{2T}=\left(h_4,h_5,h_6\right)\\
\pmb{h}^{3T}=\left(h_7,h_8,h_9\right)
\end{aligned}
$$
そうすると、ホモグラフィー変換を表す$H\pmb{x}$は次のように表される。
$$
\pmb{x}'=
H\pmb{x}=\begin{pmatrix}
\pmb{h}^{1T}\pmb{x}\\
\pmb{h}^{2T}\pmb{x}\\
\pmb{h}^{3T}\pmb{x}
\end{pmatrix}
$$
外積について、自分自身を取ると$\pmb{0}$だから、
$$
\begin{aligned}
\pmb{x}'\times\pmb{x}'&=\pmb{x}'\times H\pmb{x}'\\
&=
\begin{pmatrix}
y'\pmb{h}^{3T}\pmb{x}-\omega'\pmb{h}^{2T}\pmb{x}\\
\omega'\pmb{h}^{1T}\pmb{x}-x'\pmb{h}^{3T}\pmb{x}\\
x'\pmb{h}^{2T}\pmb{x}-y'\pmb{h}^{1T}\pmb{x}
\end{pmatrix}\\
&=0
\end{aligned}
$$
$\pmb{h}^{1T}\pmb{x}$はスカラーだから、$\pmb{x}^T\pmb{h}^{1}$と等しい。よって、例えば1行目について次のようになる。

$$
\begin{aligned}
y'\pmb{h}^{3T}\pmb{x}
-\omega'\pmb{h}^{2T}\pmb{x}
&=\pmb{0}^T\pmb{h}^1
-\omega'\pmb{x}^T\pmb{h}^2
+y'\pmb{x}^T\pmb{h}^{3}\\
&=
\begin{pmatrix}
0&0&0&
-\omega' x&-\omega' y&-\omega' \omega&
y'x&y'y&y'\omega
\end{pmatrix}
\begin{pmatrix}
\pmb{h}^{1}\\
\pmb{h}^{2}\\
\pmb{h}^{3}
\end{pmatrix}
\end{aligned}
$$
よってまとめると次のようになる。
$$
\begin{pmatrix}
\pmb{0}^T&
-\omega' \pmb{x}^T&
y' \pmb{x}^T\\
\omega' \pmb{x}^T&
\pmb{0}^T&
-x' \pmb{x}^T\\
-y' \pmb{x}^T&
x' \pmb{x}^T&
\pmb{0}^T
\end{pmatrix}
\begin{pmatrix}
\pmb{h}^{1}\\
\pmb{h}^{2}\\
\pmb{h}^{3}
\end{pmatrix}
=\pmb{0}
$$