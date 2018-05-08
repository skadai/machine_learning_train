# SVM支持向量机

本文是学习SVM的笔记，主要参考了
1. 机器学习实战
2. 知乎答案 https://zhuanlan.zhihu.com/p/24638007

有兴趣可以阅读原文


## SVM算法背景


![SVM分类描述](https://raw.githubusercontent.com/skadai/markdownphoto/master/%E6%8A%80%E6%9C%AF%E8%B5%84%E6%96%99/SVM%E5%88%86%E7%B1%BB%E6%8F%8F%E8%BF%B0.jpg)

给定平面内的两类散点，用一条直线进行分类，如何找到最好的分割线？

首先，需要给出**最好**的定义。我们希望这条线能距离两边都足够的远，这样再增加新的散点有更大的概率这条线依然是可用的。

如何描述分割线到两个集合的距离？显然是由集合中最近的点到分割线的距离决定的，这样一来问题转换为
1. 求分割线（如果是n维空间就是分割超平面的问题
2. 分割线到各个集合的最近的点的距离最大

条件2看上去十分拗口，难点也正是在此。


## 数学建模

### 目标函数

首先回顾一下点到直线的距离公式

```math
     d = \frac{|w^Tx+\gamma|}{||w||}  
```
这个公式同样适用多维情况

目标函数就是这个d的最大化问题

### 约束条件
1. 如何保证决策面？
    所有的点都刚好位于决策面的两侧？
2. 决策面应该位于间隔区域的中轴线
3. 中间的x样本点应该是最近的点，也就是支持向量


此问题可以设置label分别是`$\pm1$`
```math
y=\begin{cases}
1, \quad  case1\\
-1,\quad case2
\end{cases}
```
则有
```math
y=\begin{cases}
w^Tx+\gamma \geq1, \quad label 1\\
w^Tx+\gamma \leq-1,\quad label -1
\end{cases}
```
上面的约束条件对于支持向量等号成立,此时套用距离公式则有

```math
d=\frac{1}{||w||}
```
对所有支持向量成立

目标是d的最大化，则转换为w最小化问题
至此，数学问题可以描述为
```math
min_w,\gamma \frac{1}{2}||w||^2

s.t. \quad y_i(w^Tx_i+\gamma)\geq1 i=1,2,....m

```

## 问题的求解


1. 拉格朗日函数将约束条件耦合到目标函数
2. 拉格朗日对偶，具体证明略，通过求相对容易的对偶问题来解决目标函数的优化【步骤2其实没看懂，数学能力有限】

简单罗列下拉格朗日对偶的步骤,对一般化的优化问题

```math

min_x f(x)

s.t. \quad h_i(x)=0\quad i=1,2...m

g_i(x)\leq0\quad j=i,2,...n

```

m个等式约束和n个不等式约束

通过构造广义拉格朗日函数

```math

\theta_p(x) = max(L(x,\alpha,\beta))

L(x,\alpha,\beta) = f(x)+ \sum_{i=1}^m\alpha_ih_i(x)+\sum_{i=1}^n\beta_ig_i(x)
```

这个广义的函数满足在约束条件内 等价f(x),同时在约束条件之外等价正无穷

通过拉格朗日对偶，上面问题转换成
```math
max_{\alpha,\beta}[(\theta_D(\alpha,\beta)]=
max_{\alpha,\beta}(min_x[L(x,\alpha,\beta)])
```

上述的一般化问题对于SVM就是两个对偶问题

原始问题`$min_{w,r}[max_\alpha L(w,\gamma,\alpha)] $`

对偶问题`$max_{\alpha}[min_{\gamma,w} L(w,\gamma,\alpha)] $`

求解对偶问题，括号内
```math

min_{\gamma,w}L(\gamma,w,\alpha)=
min[\frac{1}{2}||w||^2+\sum_{i=1}^m\alpha_i(1-y_i(w^Tx_i+\gamma))]

```

对w和r求导
```math

w=\sum_{i=1}^m\alpha_iy_ix_i

0=\sum_{i=1}^m\alpha_iy_i
```
上面的结论带入L函数，然后求其最大值则最终问题形式是
```math

max_\alpha[\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_jx_i^Tx_j]

s.t.\quad \sum_{i=1}^T\alpha_iy_i=0

\alpha_i\geq0

```
