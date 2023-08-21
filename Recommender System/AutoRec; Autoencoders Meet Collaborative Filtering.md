#Recommender_System #Collaborative_Filtering #AutoEncoder

# Abstract

This paper proposes AutoRec, a novel autoencoder framework for collaborative filtering (CF).  
이 논문은 협업필터링을 위한 새로운 autoencoder 프레임워크인 AutoRec을 소개한다. 

Empirically, AutoRec’s compact and efficiently trainable model outperforms stateof-the-art CF techniques (biased matrix factorization, RBMCF and LLORMA) on the Movielens and Netflix datasets.  
경험적으로, AutoRec의 간결하고 효율적인 학습 모델은 Movielens와 Netflix 데이터셋에 대하여 현재의 SOTA인 협업필터링 기법(biased MF, RBMCF, LLORMA)을 뛰어 넘는 성능을 보여준다.  

<br>

# 1. Introduction
Collaborative filtering (CF) models aim to exploit information about users’ preferences for items (e.g. star ratings) to provide personalised recommendations.  
협업 필터링 모델들은 개인화된 추천을 제공하기 위해 아이템에 대한 유저의 선호도를 추출해 내는 것에 초점을 맞추고 있다.  

Owing to the Netflix challenge, a panoply of different CF models have been proposed, with popular choices being matrix factorisation and neighbourhood models.  
Netflix 경진대회 때문에, matrix factorization, neighborhood기반 모델과 같은 여러 종류의 협업 필터링 모델들이 제안되었다.  

This paper proposes AutoRec, a new CF model based on the autoencoder paradigm; our interest in this paradigm stems from the recent successes of (deep) neural network models for vision and speech tasks.  
본 논문은 autoencoder를 기반한 새로운 협업필터링 패러다임인 AutoRec을 제안한다; 우리의 패러다임은 최근 딥러닝(인공신경망)을 사용하여 비전과 음성 분야에서 큰 성공을 보여준 것에서 출발하였다.  

We argue that AutoRec has representational and computational advantages over existing neural approaches to CF, and demonstrate empirically that it outperforms the current state-of-the-art methods.  
우리는 신경망을 사용하여 협업필터링에 적용한 기존 모델들과는 달리 표현력도 뛰어나고 계산효율성도 좋은 AutoRec을 소개한다. 그리고 AutoRec을 현재 SOTA 방법들과 비교해 실험적으로 성능이 더 좋음을 보일 것이다.  

<br>

# 2. The Autorec Model

In rating-based collaborative filtering, we have $m$ users, $n$ items, and a partially observed user-item rating matrix $R \in \mathbb{R}^{m\times n}$.  
평점 데이터를 사용하는 협업필터링은 $m$명의 유저와 $n$개의 상품 그리고 일부가 채워져있는 sparse한 평점 행렬 $R \in \mathbb{R}^{m\times n}$ 이 존재한다.  

Each user $u \in U = \{1 . . . m\}$ can be represented by a partially observed vector $\mathbf{r}^{(u)} = (R_{u1}, . . . R_{un}) \in \mathbb{R}^n$ .  
각 유저 $u$는 평점행렬을 사용하여 하나의 벡터로 표현이 가능하다: $\mathbf{r}^{(u)} = (R_{u1}, . . . R_{un}) \in \mathbb{R}^n$.  

Similarly, each item $i \in I = \{1 . . . n\}$ can be represented by a partially observed vector $\mathbf{r}^{(i)} = (R_{1i}, . . . R_{mi}) \in \mathbb{R}^m$.  
마찬가지로, 각 아이템 $i$도 평점행렬을 사용하여 하나의 벡터로 표현이 가능하다: $\mathbf{r}^{(i)} = (R_{1i}, . . . R_{mi}) \in \mathbb{R}^m$.  

Our aim in this work is to design an item-based (user-based) autoencoder which can take as input each partially observed $\mathbf{r}^{(i)}\left(\mathbf{r}^{(u)}\right)$, project it into a low-dimensional latent (hidden) space, and then reconstruct $\mathbf{r}^{(i)}\left(\mathbf{r}^{(u)}\right)$ in the output space to predict missing ratings for purposes of recommendation.  
우리 연구의 목적은 아이템 기반(또는 유저 기반) autoencoder로서, 입력인 아이템 벡터(또는 유저벡터)를 저차원의 잠재 공간으로 투영시킨 후, 다시 입력으로 들어갔던 아이템 벡터(또는 유저벡터)를 복원하여 아직 평점이 없는 데이터를 예측하는 것이다.  

Formally, given a set $\mathbf{S}$ of vectors in $\mathbb{R}^d$ , and some $k \in \mathbb{N}\_{+}$, an autoencoder solves:  
수식으로 표현하자면, $d$ 차원을 갖는 벡터의 집합 $\mathbf{S}$와 $k$를 사용하여 오토인코더는 다음 문제를 해결한다:  
$$\tag{1} \min_\theta \sum_{\mathbf{r} \in \mathbf{S}} \| \mathbf{r} - h(\mathbf{r}; \theta) \|_2^2$$

where $h(\mathbf{r}; \theta)$ is the reconstruction of input $\mathbf{r} \in \mathbb{R}^d$ ,  
여기서 $h(\mathbf{r}; \theta)$은 입력 $\mathbf{r} \in \mathbb{R}^d$를 복원하는 함수를 말한다.  
$$h(\mathbf{r}; \theta) = f(\mathbf{W}\cdot g(\mathbf{Vr} + \mu) + \mathbf{b})$$
for activation functions $f(\cdot), g(\cdot)$.  
$f(\cdot), g(\cdot)$는 활성함수를 가리킨다.  

Here, $\theta = \{\mathbf{W, V}, µ, b\}$ for transformations $\mathbf{W} \in \mathbb{R}^{d×k} , \mathbf{V}\in \mathbb{R}^{k×d}$, and biases $\mu \in \mathbb{R}^k , b \in \mathbb{R}^d$.  
위 수식에서 등장하는 파라미터 집합 $\theta$에서 $\mathbf{W, V}$는 선형변환을 위한 것이고 $\mu, b$는 bias term을 의미한다.  

This objective corresponds to an auto-associative neural network with a single, $k$-dimensional hidden layer.   
위의 구조는 $k$차원의 hidden layer를 갖는 신경망에 해당한다.  

The parameters $\theta$ are learned using backpropagation.  
위에서 언급했던 파라미터 집합 $\theta$는 역전파를 통하여 학습된다.

![[AutoRec_Figure1.png| 800]]

The item-based AutoRec model, shown in Figure 1, applies an autoencoder as per Equation 1 to the set of vectors $\{\mathbf{r}(i)\}_{i=1}^n$, with two important changes.  
Figure1은 아이템을 기반으로한 AutoRec 모델을 나타낸다.  모델은 Equation 1과 같은 오토인코더 구조를 $n$개의 벡터집합에 적용하며 2가지 중요한 변화들이 존재한다.

First, we account for the fact that each $\mathbf{r}(i)$ is partially observed by only updating during backpropagation those weights that are associated with observed inputs, as is common in matrix factorisation and RBM approaches.  
첫 번째로 matrix factorization이나 RBM 접근 방식에서 일반적으로 볼 수 있듯이 아이템의 rating 벡터 $\mathbf{r}^{(i)}$는 역전파 중에 관측된 입력과 관련된 가중치만 업데이트하는 사실을 설명할 것이다.

Second, we regularise the learned parameters so as to prevent overfitting on the observed ratings.
두 번째로, 우리는 학습된 파라미터에 규제를 두어 관측된 평점 값에 과적합되는 것을 예방할 것이다.

Formally, the objective function for the Item-based AutoRec (I-AutoRec) model is, for regularisation strength λ > 0,  
규제 정도 $\lambda > 0$ 에 대해 아이템을 기반으로 한 AutoRec을 수식으로 나타내자면, 다음과 같다.

$$\tag{2} \min_\theta \sum_{i=1}^n \| \mathbf{r}^{(i)} - h(\mathbf{r}^{(i)}; \theta) \|_\mathcal{O}^2 + \frac{\lambda}{2} \cdot (\| \mathbf{W}\|_F^2 + \|\mathbf{V}\|_F^2)$$
where $\|\cdot\|_\mathcal{O}^2$ means that we only consider the contribution of observed ratings.  
$\|\cdot\|_\mathcal{O}^2$이 말하는 것은 오직 관측된 평점 데이터에 대해서만 학습을 진행한다는 의미이다.

User-based AutoRec (U-AutoRec) is derived by working with $\{\mathbf{r}(u)\}_{u=1}^m$.  
유저를 기반으로한 AutoRec은 유저 평점 벡터 $\{\mathbf{r}(u)\}_{u=1}^m$ 를 사용한다.

In total, I-AutoRec requires the estimation of $2mk + m + k$ parameters.  
종합적으로 I-AutoRec은 추정을 하기 위해 모델에서 $2mk + m + k$만큼의 파라미터를 필요로 한다.

>[!info] Parameter 수 (아이템을 기반으로 한 AutoRec)
> - $\mathbf{W}\in\mathbb{R}^{m \times k}$ : $mk$ 개의 파라미터
> - $\mathbf{V}\in\mathbb{R}^{k \times m}$ : $mk$ 개의 파라미터
> - $\mu\in\mathbb{R}^{k}$ : $k$ 개의 파라미터
> - $\mathbf{b}\in\mathbb{R}^{m}$ : $m$ 개의 파라미터

Given learned parameters $\hat{\theta}$, I-AutoRec’s predicted rating for user $u$ and item $i$ is  
학습된 파라미터 $\hat{\theta}$ 를 사용한다면, I-AutoRec이 예측한 유저 $u$의 상품 $i$에 대한 평점은 다음과 같다.

$$\tag{3} \hat{R}_{ui} = (h(\mathbf{r}^{(i)}; \hat{\theta}))_u$$
Figure 1 illustrates the model, with shaded nodes corresponding to observed ratings, and solid connections corresponding to weights that are updated for the input $\mathbf{r}^{(i)}$.  
Figure 1이 해당 모델을 묘사하는 그림이며, 어둡게 칠해진 노드가 관측된 평점, 실선으로 연결된 것이 입력 $\mathbf{r}^{(i)}$로 업데이트 되는 가중치를 말한다.

AutoRec is distinct to existing CF approaches.  
AutoRec은 기존에 존재하는 협업필터링 접근방식과는 구분된다.

Compared to the RBM-based CF model (RBM-CF), there are several differences.  
AutoRec은 RBM기반 협업필터링과 비교하였을 때, 몇 가지 차이점이 존재한다.

First, RBM-CF proposes a generative, probabilistic model based on restricted Boltzmann machines, while AutoRec is a discriminative model based on autoencoders.  
첫 번째로 RBM-CF는 restricted Boltzmann machine을 기반으로 하여 일반화된 확률 모델을 제안하였지만, AutoRec은 autoencoder를 기반으로 한 구별되는 모델이다.

Second, RBM-CF estimates parameters by maximising log likelihood, while AutoRec directly minimises RMSE, the canonical performance in rating prediction tasks.  
두 번째로 RBM-CF는 log likelihood를 최대화하며 파라미터를 추정하는 반면 AutoRec은 평점예측 문제의 표준이라고 볼 수 있는 RMSE를 바로 최소화하는 방식을 사용하였다.

Third, training RBM-CF requires the use of contrastive divergence, whereas training AutoRec requires the comparatively faster gradient-based backpropagation.  
세 번재로 RBM-CF를 학습할 때 contrastive divergence를 필요로 하였지만, AutoRec을 학습시킬 때는 그래디언트 기반 역전파를 사용하여 훨씬 빠른 속도로 학습할 수 있었다.

Finally, RBM-CF is only applicable for discrete ratings, and estimates a separate set of parameters for each rating value.  
마지막으로 RBM-CF는 오직 이산적인 평점에 대해서만 적용가능하며, 각 평점에 대한 별도의 매개 변수 집합을 추정하고 있다.

For $r$ possible ratings, this implies $nkr$ or ($mkr$) parameters for user- (item-) based RBM.  
만약 $r$개의 가능한 평점이 존재한다면 이는 $nkr$개 (또는 $mkr$) 개의 파라미터가 유저(아이템) 기반 RBM에서 필요로 한다.

AutoRec is agnostic to $r$ and hence requires fewer parameters.  
AutoRec 은 평점의 개수 $r$에 상관없이 동작하며 이는 더 적은 양의 파라미터를 갖게 만들어준다.

Fewer parameters enables AutoRec to have less memory footprint and less prone to overfitting.  
적은 크기의 파리미터를 갖는 AutoRec 덕분에 더 적은 메모리를 사용할 수 있게 되고 과적합을 하는 경향도 줄어든다.

Compared to matrix factorisation (MF) approaches, which embed both users and items into a shared latent space, the item-based AutoRec model only embeds items into latent space.  
잠재공간으로 유저와 아이템 모두 임베딩하는 Matrix factorization 기법과 비교하였을 때, 아이템 기반 AutoRec 모델은 오직 아이템만 잠재공간으로 임베딩한다.

Further, while MF learns a linear latent representation, AutoRec can learn a nonlinear latent representation through activation function $g(\cdot)$.  
게다가 Matrix Factorization은 선형 잠재 표현도 학습을 시켜야 하지만, AutoRec은 활성함수 $g(\cdot)$를 통해 비선형 잠재 변환을 학습한다.

<br>

# 3. Experimental Evaluation

In this section, we evaluate and compare AutoRec with RBM-CF, Biased Matrix Factorisation (BiasedMF), and Local Low-Rank Matrix Factorisation (LLORMA) on the Movielens 1M, 10M and Netflix datasets.  
이번 섹션에서 우리는 Movielens 1M, 10M 그리고 넷플릭스 데이터셋에 대해 AutoRec이 다른 모델 (RBM-CF, BiasedMF, LLORMA) 대비 성능 비교를 진행할 것이다.

Following LLORMA, we use a default rating of 3 for test users or items without training observations.  
LLORMA 논문에 이어 우리는 3명의 테스트 유저를 따로 분리하였다. (3명의 유저에 대해서는 학습을 진행하지 않음)

We split the data into random 90%–10% train-test sets, and hold out 10% of the training set for hyperparamater tuning.  
우리는 데이터를 학습 90%, 테스트 10%로 분리하였으며 학습 데이터셋의 10%를 다시 하이퍼파라미터를 튜닝하기 위해 사용하였다.

We repeat this splitting procedure 5 times and report average RMSE.  
우리는 5번을 반복하여 분리하는 절차를 걸쳤고, 평균 RMSE 점수를 계산하였다.

95% confidence intervals on RMSE were $\pm 0.003$ or less in each experiment.  
각 실험에서 RMSE의 95% 신뢰구간은 $\pm 0.003$ 이하의 변동값을 가지고 있었다. 

For all baselines, we tuned the regularisation strength $\lambda \in \{0.001, 0.01, 0.1, 1, 100, 1000\}$ and the appropriate latent dimension $k \in \{10, 20, 40, 80, 100, 200, 300, 400, 500\}$.  
각각의 baseline 모델들에 대해 우리는 규제 강도 $\lambda$ 와 잠재 공간의 차원 $k$에 대해 하이퍼파라미터 튜닝을 진행했다.

A challenge training autoencoders is non-convexity of the objective.  
Autoencoder 모델을 학습시키는 가장 큰 문제는 ==목적함수가 항상 convex하지 않다는 것==이다.

We found resilient propagation (RProp) to give comparable performance to L-BFGS, while being much faster.  
우리는 L-BFGS보다 성능이 좋은 resilient propagation을 발견하였다.

Thus, we use RProp for all subsequent experiments:  
따라서 우리는 RProp을 모든 후속 실험에 적용하였다.


Which is better, item- or user-based autoencoding with RBMs or AutoRec?  
RBM이나 AutoRec을 사용하는데 있어서 아이템 기반 autoencoding이 좋을까 아니면 유저 기반 autoencoding이 좋을까?

![[AutoRec_Table1.png]]

Table 1a shows item-based (I-) methods for RBM and AutoRec generally perform better;  
_표1a는  아이템을 기반으로 한 RBM과 AutoRec이 일반적으로 성능이 좋다는 것을 보인다._

this is likely since the average number of ratings per item is much more than those per user;  
_이것은 아마 아이템에 매겨진 평점이 유저가 매긴 평점의 수보다 많기 때문일 것이다._

high variance in the number of user ratings leads to less reliable prediction for user-based methods.  
_유저 평점의 높은 편차들이 유저 기반 데이터로 학습한 모델로 예측했을 때, 신뢰도를 떨어트리는 경향이 있기 때문이다._

I-AutoRec outperforms all RBM variants.  
_또한, 모든 결과에서 I-AutoRec이 RBM보다 더 좋은 성능을 보여주었다._

How does AutoRec performance vary with linear and nonlinear activation functions $f(\cdot), g(\cdot)$?  
AutoRec의 성능이 선형 활성함수와 비선형 활성함수에 따라 어떻게 변화하는가?

Table 1b indicates that nonlinearity in the hidden layer (via $g(\cdot)$) is critical for good performance of I-AutoRec, indicating its potential advantage over MF methods.  
표1b는 I-AutoRec에서 비선형 활성함수를 사용하여 hidden layer에 비선형성을 준 것이 성능에 중대한 영향을 줌을 확인하였다. 이것은 모든 Matrix factorization 기반 방법에서 동일한 결과가 나왔다.

Replacing sigmoids with Rectified Linear Units (ReLU) performed worse.  
비선형 함수들 중에서는 sigmoid함수를 ReLU로 교체했을 때 성능이 나빠졌다.

All other AutoRec experiments use identity $f(\cdot)$ and sigmoid $g(\cdot)$ functions.  
모든 AutoRec 실험은 선형함수로 identity ($y=x$)를 사용하였고, 비선형함수로는 sigmoid를 사용하였다.

How does performance of AutoRec vary with the number of hidden units?  
hidden unit의 수가 변화함에 따라 AutoRec의 성능은 어떻게 변화하는가?

![[AutoRec_Figure2.png]]

In Figure 2, we evaluate the performance of AutoRec model as the number of hidden units varies.  
Figure2에서 우리는 hidden unit의 수의 변화에 따라 AutoRec 모델의 성능은 어떻게 변화하는지 평가해보았다.

We note that performance steadily increases with the number of hidden units, but with diminishing returns.  
우리는 hidden unit의 수를 증가시킬 수록 성능은 증가하지만 점차 증가폭이 줄어든다는 것을 확인하였다.

All other AutoRec experiments use $k = 500$.  
다른 모든 AutoRec 실험에서는 $k=500$을 사용하였다.

How does AutoRec perform against all baselines?  
AutoRec이 다른 baseline 모델들과 비교하였을 때 성능이 어떠한가?

Table 1c shows that AutoRec consistently outperforms all baselines, except for comparable results with LLORMA on Movielens 10M.  
표1c는 AutoRec이 일관되게 다른 모든 baseline에 비해 좋은 성능을 보여주었으며, Movielens 10M 데이터셋에 대해서는 LLORMA와 동일한 성능을 보여주었다.

Competitive performance with LLORMA is of interest, as the latter involves weighting 50 different local matrix factorization models, whereas AutoRec only uses a single latent representation via a neural net autoencoder.  
LLORMA는 local matrix factorization 모델에 50개의 서로 다른 가중치를 부여하였지만, AutoRec은 auto encoder를 사용하여 오직 하나의 잠재 표현만 사용하였는데도 동일하게 나온 것은 흥미로웠다.

Do deep extensions of AutoRec help?  
AutoRec의 깊이를 확장시키면 도움이 될까?

We developed a deep version of I-AutoRec with three hidden layers of (500, 250, 500) units, each with a sigmoid activation.  
우리는 I-AutoRec을 3개의 레이어로 늘려 성능을 측정해보았다. 각 레이어에는 500, 250, 500개의 hidden unit이 있도록 설정했고 활성함수로는 sigmoid를 사용하였다.

We used greedy pretraining and then fine-tuned by gradient descent.  
우리는 pretrain을 먼저 진행하였고 이후 경사하강법으로 fine tuning하는 방법을 사용하였다.

On Movielens 1M, RMSE reduces from 0.831 to 0.827 indicating potential for further improvement via deep AutoRec.  
Movielens 1M 데이터셋에 대해 RMSE 결과는 0.831에서 0.827로 감소하였으며, 이는 더 깊은 레이어를 쌓아 개선된 AutoRec을 만들 수 있는 잠재력을 보여준다.
