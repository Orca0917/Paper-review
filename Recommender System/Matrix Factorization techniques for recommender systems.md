#Collaborative_Filtering #Matrix_Factorization #Recommender_System


# Introduction

As the Netflix Prize competition has demonstrated, matrix factorization models are superior to classic nearest-neighbor techniques for producing product recommendations, allowing the incorporation of additional information such as implicit feedback, temporal effects, and confidence levels.
_Netflix Prize 경진대회에서 입증했듯이, Matrix Factorization 모델은 implicit feedback이나 시간적 요소, 신뢰도 레벨과 같은 추가 정보를 사용하여 이전 아이템 추천을 해주는 Nearest-neighbor 기법보다 더 우수한 성적을 보여주었다._

Modern consumers are inundated with choices.
_현대의 소비자들은 선택의 늪에 빠져있다._

Electronic retailers and content providers offer a huge selection of products, with unprecedented opportunities to meet a variety of special needs and tastes.
_전자상거래 판매자들과 컨텐츠 제공자들은 소비자들이 다양한 욕구와 입맛에 맞게 상품을 선택할 수 있게 하고 있다._

Matching consumers with the most appropriate products is key to enhancing user satisfaction and loyalty.
_판매자들은 소비자들이 원하는 상품들을 연결시켜 줌으로써 사용자들의 만족감과 충성심을 높일 수 있다._

Therefore, more retailers have become interested in recommender systems, which analyze patterns of user interest in products to provide personalized recommendations that suit a user’s taste.
_그러므로, 많은 판매자들은 유저의 소비 패턴을 분석해서 좋아할만한 아이템을 추천해주는 추천시스템에 관심을 갖게 되었다._

Because good personalized recommendations can add another dimension to the user experience, e-commerce leaders like [Amazon.com](http://amazon.com/) and Netflix have made recommender systems a salient part of their websites.
_좋은 추천을 만들면 유저에게 새로운 차원의 경험을 하게 만들 수 있기에, Amazon이나 Netflix와 같은 E-커머스의 리더들은 자신들의 웹사이트에 추천시스템을 탑재하는 것을 가장 중요한 요소로 뽑았다._

Such systems are particularly useful for entertainment products such as movies, music, and TV shows.
_이런 추천시스템은 특히 영화, 음악, TV쇼와 같은 엔터테인먼트 컨텐츠에 더 유용하다._

Many customers will view the same movie, and each customer is likely to view numerous different movies.
_많은 고객들은 동일한 영화를 볼 것이고, 각 고객들은 다양한 또 다른 영화들을 좋아한다._

Customers have proven willing to indicate their level of satisfaction with particular movies, so a huge volume of data is available about which movies appeal to which customers.
_사람들은 특정 영화에 대해서만 만족감을 크게 느끼기 때문에, 우리는 어떤 사람들이 무슨 영화를 시청하고 또 좋아했는지에 대한 방대한 데이터를 획득할 수 있다._

Companies can analyze this data to recommend movies to particular customers.
_회사들은 이런 데이터들을 분석하여 특정 고객들을 대상으로 영화를 추천한다._

<br>

# Recommender System Strategies

Broadly speaking, recommender systems are based on one of two strategies.
_대체로, 추천시스템들은 2가지 전략 중 하나를 기반으로 하고 잇다._

The content filtering approach creates a profile for each user or product to characterize its nature.
_컨텐츠 필터링 접근법은 각 유저나 아이템별로 그 특징들을 표현하는 프로필을 생성한다._

For example, a movie profile could include attributes regarding its genre, the participating actors, its box office popularity, and so forth.
_예를 들어, 영화 프로필은 어떤 장르인지, 어떤 배우들이 출현하는지, 박스오피스 흥행도는 어떤지와 같은 정보들을 담을 수 있다._

User profiles might include demographic information or answers provided on a suitable questionnaire.
_유저 프로필은 인구통계학적 정보나 몇 가지 설문에 관련된 대답 정보들을 담고 있을 수 있다._

The profiles allow programs to associate users with matching products.
_이 프로필들을 사용하여 프로그램은 유저와 알맞는 상품(아이템)을 매칭시킨다._

Of course, content-based strategies require gathering external information that might not be available or easy to collect.
_물론, 컨텐츠 기반 전략을 사용하게 되면, 사용하기 어렵거나 수집하기가 까다로운 정보들을 요구할 수 있다._

A known successful realization of content filtering is the Music Genome Project, which is used for the Internet radio service [Pandora.com](http://pandora.com/).
_컨텐츠 필터링 방법을 가장 잘 실체화 한 것은 Music Genome 프로젝트이며 이는 현재 Pandora.com에서 인터넷 라디오 서비스에 사용되고 있다._

A trained music analyst scores each song in the Music Genome Project based on hundreds of distinct musical characteristics.
_학습된 음악 분석기는 Music Genome Project에 존재하는 각 음악에 대해 점수를 매겼다._

These attributes, or genes, capture not only a song’s musical identity but also many significant qualities that are relevant to understanding listeners’ musical preferences.
_이런 속성이나 장르는 음악의 음악적 특색 뿐만 아니라 청취자들의 음악 선호 취향을 이해하는 것과 관련된 중요한 품질들도 포착한다._

An alternative to content filtering relies only on past user behavior for example, previous transactions or product ratings without requiring the creation of explicit profiles.
_컨텐츠 필터링의 대안은 사용자들의 과거 행동기록에 의존한다. 예를 들어, 이전의 구매기록이나 남긴 평점 기록들이 있으며, 이런 방식의 장점은 별도의 프로필 생성이 불필요하다는 것이다._

This approach is known as collaborative filtering, a term coined by the developers of Tapestry, the first recommender system.1
_이 방식은 협업 필터링(Collaborative Filtering)이라고도 알려져 있다._

Collaborative filtering analyzes relationships between users and interdependencies among products to identify new user-item associations.
_협업 필터링은 이미 존재하는 유저와 아이템들의 관계를 분석하여, 새로운 유저와 아이템 관계가 주어졌을 때 선호도를 예측한다._

A major appeal of collaborative filtering is that it is domain free, yet it can address data aspects that are often elusive and difficult to profile using content filtering.
_협업필터링의 주요 장점은 특정 도메인 지식이 필요하지 않다는 것이다. 따라서 데이터를 다룰 때, 프로필을 생성하기 어렵거나 애매한 경우에도 사용 가능하다._

While generally more accurate than content-based techniques, collaborative filtering suffers from what is called the cold start problem, due to its inability to address the system’s new products and users.
_일반적으로 컨텐츠 기반 기법에 비해 더 높은 정확도를 보이는 반면, 협업필터링은 Cold-start 문제에 대해 고통 받고 있다. (시스템에 새로운 상품이나 유저가 들어왔을 때 기존의 정보가 없어서 다룰 수 없음)_

In this aspect, content filtering is superior.
_이런 측면에서 보았을 때는 컨텐츠 필터링이 더 우세하다._

The two primary areas of collaborative filtering are the neighborhood methods and latent factor models.
_협업필터링 안에서도 2가지 주된 분야가 있는데, 각각 이웃기반방법(neighborhood methods)과 잠재요소(latent factor)모델이다._

Neighborhood methods are centered on computing the relationships between items or, alternatively, between users.
_이웃기반 방법은 아이템간의 관계나 유저간의 관계들을 포착하는데 더 집중한다._

The item oriented approach evaluates a user’s preference for an item based on ratings of “neighboring” items by the same user.
_아이템을 기반으로한 방법은 유저의 선호도를 아이템을 기반으로 측정하며, 동일한 유저가 선호했던 다른 아이템들을 참조한다._

A product’s neighbors are other products that tend to get similar ratings when rated by the same user.
_아이템(상품)의 이웃이라고 하는 것은 동일한 유저에 의해 비슷한 평점이 매겨진 다른 아이템들을 말한다._

For example, consider the movie Saving Private Ryan.
_예를 들어, 영화 ‘라이언 일병 구하기’를 고려해보자._

Its neighbors might include war movies, Spielberg movies, and Tom Hanks movies, among others.
_해당 영화의 이웃들은 아마 전쟁 영화이거나, 스필버그 감독이 만든 영화이거나 톰 행크스가 출현한 영화일 수 있다._

To predict a particular user’s rating for Saving Private Ryan, we would look for the movie’s nearest neighbors that this user actually rated.
_특정 유저가 라이언 일병 구하기에 내릴 평점을 예측하기 위해서 우리는 그 유저가 실제로 평점을 내린 다른 이웃 영화들을 참고해야한다._


![[MF_Figure1.png| 600]]


As Figure 1 illustrates, the user-oriented approach identifies like-minded users who can complement each other’s ratings.
_Figure 1이 보여주듯이 유저기반 접근법은 비슷한 취향을 가진 유저들을 찾아 서로의 평점을 예측하는데 사용한다._

Latent factor models are an alternative approach that tries to explain the ratings by characterizing both items and users on, say, 20 to 100 factors inferred from the ratings patterns.
_잠재요소 모델은 평점 패턴으로 부터 아이템이나 유저의 특징을 20~100개의 잠재 요소들로 추론하고 평점을 예측하는 다른 대안이다._

In a sense, such factors comprise a computerized alternative to the aforementioned humancreated song genes.
_어떤 의미에서는 20~100개의 요소들은 앞서 언급한 음악의 장르와 같은 특징들을 컴퓨터로 계산하고 압축된 형태로 해석할 수 있다._

For movies, the discovered factors might measure obvious dimensions such as comedy versus drama, amount of action, or orientation to children; less well-defined dimensions such as depth of character development or quirkiness; or completely uninterpretable dimensions.
_영화에 대해서 말을 하자면, 찾아낸 요소들은 코미디나 드라마의 장르를 가리킬 수도 있고, 등장하는 액션의 수, 아이들을 대상으로 한 영화인지와 같은 깔끔하게 분류된 속성들일 수 있다. 잘 정의되지 않은 차원들이 학습될 수도 있으며 완전히 우리가 해석할 수 없는 차원을 가리킬 수도 있다._

For users, each factor measures how much the user likes movies that score high on the corresponding movie factor.
_유저에 대해서 각 요소는 유저가 영화를 얼마나 좋아할 것인지를 측정한다._


![[MF_Figure2.png | 600]]


Figure 2 illustrates this idea for a simplified example in two dimensions.
_Figure 2는 2개의 차원을 예시로 든 간단한 이미지이다._

Consider two hypothetical dimensions characterized as female versus male oriented and serious versus escapist
_예를 들어 2개의 임의의 차원이 <성별>, <진지함> 의 속성을 나타낸다고 고려해보자._

The figure shows where several well-known movies and a few fictitious users might fall on these two dimensions.
_사진은 몇 개의 잘 알려진 영화들과 몇 명의 임의의 유저를 2차원 공간에 표현하였다._

For this model, a user’s predicted rating for a movie, relative to the movie’s average rating, would equal the dot product of the movie’s and user’s locations on the graph.
_이 모델에 대해서, 영화의 평균 평점과 관련해서 유저가 영화에 내릴 평점은 그래프 위에서 유저의 위치와 아이템의 위치를 내적한 것과 동일하다._


> [!info] 예측한 평점이 유저벡터와 아이템 벡터를 내적한 것과 같은 이유: 비슷한 곳에 위치할 수록 더 큰 값을 갖는 내적의 특성 때문


For example, we would expect Gus to love Dumb and Dumber, to hate The Color Purple, and to rate Braveheart about average.
_위의 예제에서는 Gus가 덤앤더머 영화를 좋아할 것이고 Color Purple 영화를 싫어하며, Braveheart 에 대해서는 평균 평점을 내릴 것으로 볼 수 있다._

Note that some movies for example, Ocean’s 11 and users for example, Dave would be characterized as fairly neutral on these two dimensions.
_영화 오션스11와 유저 Dave는 현재 2개의 차원 위에서는 공평하게 중립을 유지함을 보이는 것에 주목할 필요가 있다._

<br>

# Matrix Factorization Methods

Some of the most successful realizations of latent factor models are based on matrix factorization.
_몇 가지 성공적인 latent factor model들은 matrix factorization에 기반을 두고 있다._

In its basic form, matrix factorization characterizes both items and users by vectors of factors inferred from item rating patterns. 
_기본적인 형태로, matrix factorization은 아이템 평점 패턴으로 부터, 아이템 잠재벡터와 유저 잠재벡터를 통해 특징들을 추론해낸다._

High correspondence between item and user factors leads to a recommendation. 
_아이템 잠재벡터와 유저 잠재벡터 사이의 높은 일치는 추천으로 이어지게 된다._

These methods have become popular in recent years by combining good scalability with predictive accuracy.
_이런 방법들은 최근 몇 년간 뛰어난 확장성과 예측 정확도를 보여주었기에 매우 유명해지게 되었다._

In addition, they offer much flexibility for modeling various real-life situations. 
_추가로, 이 방법들은 실생활의 다양항 상황을 모델링하는데 있어서 매우 유연함을 제공하였다._

Recommender systems rely on different types of input data, which are often placed in a matrix with one dimension representing users and the other dimension representing items of interest.
_추천시스템은 여러가지 종류의 입력데이터에 의존하고 있으며, 일반적으로 2차원 형태의 행렬로 주어진다. 하나의 축은 유저를 나타내고 다른 하나의 축은 아이템을 나타낸다._

The most convenient data is high-quality explicit feedback, which includes explicit input by users regarding their interest in products.
_사용하기에 가장 편리한 데이터는 높은 품질을 갖는 explicit feedback 데이터이며, 해당 데이터는 유저가 특정 상품을 얼마나 좋아하는지 직접 입력한 값이다._

For example, Netflix collects star ratings for movies, and TiVo users indicate their preferences for TV shows by pressing thumbs-up and thumbs-down buttons.
_예를 들어, Netflix는 영화에 대한 평점을 수집하며, TiVo의 유저들은 TV Show에 대한 선호도를 좋아요 또는 싫어요 버튼을 통해 구분하게 된다._

We refer to explicit user feedback as ratings.
_우리는 Explicit user feedback을 평점으로 간주한다._

Usually, explicit feedback comprises a sparse matrix, since any single user is likely to have rated only a small percentage of possible items.
_일반적으로 explicit feedback 데이터는 희소한 행렬 형태를 띈다. (한명의 유저가 다른 모든 영화들을 시청하거나 구매하지는 않기 때문 - 극히 일부만 시청하거나 구매)_

One strength of matrix factorization is that it allows incorporation of additional information. 
_Matrix factorization의 한 가지 장점은 추가 정보의 사용을 허용한다는 것이다._

When explicit feedback is not available, recommender systems can infer user preferences using implicit feedback, which indirectly reflects opinion by observing user behavior including purchase history, browsing history, search patterns, or even mouse movements.
_만약 explicit feedback을 수집하거나 사용하지 못한다면, 추천시스템은 implicit feedback을 통해서 사용자의 선호도를 추론한다. (유저의 시청기록, 유저의 구매 행동, 검색 기록, 마우스 이벤트들을 통해 추론)_

Implicit feedback usually denotes the presence or absence of an event, so it is typically represented by a densely filled matrix.
_Implicit feedback은 일반적으로 어떤 특정 이벤트(시청, 구매)를 나타내므로 행렬을 거의 가득 채워진 형태로 보여진다 (희소하지 않음)_

<br>

# A basic matrix factorization model

Matrix factorization models map both users and items to a joint latent factor space of dimensionality $f$, such that user-item interactions are modeled as inner products in that space.
_Matrix factorization은 유저와 아이템 모두 $f$ 차원을 갖는 잠재공간(joint latent space)에 매핑하여, 유저-아이템 상호작용을 잠재공간에서의 내적으로 모델링한다._

Accordingly, each item $i$ is associated with a vector $q_i \in \mathbb{R}^f$ , and each user $u$ is associated with a vector $p_u \in \mathbb{R}^f$ .
_따라서, 각 아이템 $i$ 는 벡터 $q_i \in \mathbb{R}^f$ 로 표현되며 유저 $u$ 는 벡터 $p_u \in \mathbb{R}^f$ 로 표현한다._

For a given item $i$, the elements of $q_i$ measure the extent to which the item possesses those factors, positive or negative. 
_주어진 아이템 $i$ 에 대해서, $q_i$ 의 원소들이 말하는 것은 $f$ 개의 잠재요소에 대해 positive 한지, negative 한지 나타낸다._

For a given user $u$, the elements of $p_u$ measure the extent of interest the user has in items that are high on the corresponding factors, again, positive or negative. 
_어떤 유저 $u$ 에 대해 $p_u$ 의 원소는 아이템과 마찬가지로 여러 아이템들에 대한 잠재요소들 (ex. 전자기기, 색상 등..)을 좋아하는지 싫어하는지의 값을 나타낸다._

The resulting dot product, $q_i ^\top p_u$ , captures the interaction between user $u$ and item $i$  -the user’s overall interest in the item’s characteristics.
_두 잠재벡터의 내적 $q_i^\top p_u$ 는 유저 $u$와 아이템 $i$ 사이의 상호작용을 계산한다._

This approximates user $u$’s rating of item $i$, which is denoted by $r_{ui}$, leading to the estimate $$\tag{1} \hat{r}_{ui} = q_i^\top p_u$$
*위의 계산된 값은 유저 $u$가 아이템 $i$에 내릴 평점인 $r_{ui}$를 의미하며, 이것이 실제 평점을 추정하도록 만들어야 한다.*

The major challenge is computing the mapping of each item and user to factor vectors $q_i, p_u \in \mathbb{R}^f$.
*이제 주요 과제는 유저 잠재벡터와 아이템 잠재벡터를 학습하는 것이다.*

After the recommender system completes this mapping, it can easily estimate the rating a user will give to any item by using Equation 1.
*만약 추천시스템이 각 잠재벡터를 완전하게 잘 학습했다면, 이후 평점예측은 아주 간단하게 내적으로 계산할 수 있다.*

Such a model is closely related to singular value decomposition (SVD), a well-established technique for identifying latent semantic factors in information retrieval.
_위 모델은 Singular Value Decomposition (SVD)와 상당히 밀접한 관계를 갖고 있으며, SVD는 정보 추출 분야에서 잠재 요인을 식별하는 기술이다._

Applying SVD in the collaborative filtering domain requires factoring the user-item rating matrix. 
_SVD를 협업 필터링 도메인에 적용하는 것은 유저-아이템 평점 행렬을 분해하는 과정이 필요하다._

This often raises difficulties due to the high portion of missing values caused by sparseness in the user-item ratings matrix. 
_위의 분해하는 과정은 행렬에 너무 많은 결측치가 존재하기에 어려움이 존재한다._

Conventional SVD is undefined when knowledge about the matrix is incomplete. 
_전통적인 SVD방식은 이와 같은 행렬에 결측값이 매우 많이 존재한다면 정의되지 않는다._

Moreover, carelessly addressing only the relatively few known entries is highly prone to overfitting. 
_그렇다고, 값이 존재하는 entry에 대해서만 학습을 진행하게 되면 overfitting이 될 확률이 높아진다._

Earlier systems relied on imputation to fill in missing ratings and make the rating matrix dense.
_그래서 일반적인 초기의 머신러닝 모델들은 dense한 행렬을 만들기 위해 결측값들을 imputation 해주었다._

However, imputation can be very expensive as it significantly increases the amount of data. 
_하지만, 데이터의 양이 늘어날 수록 imputation을 하는데 드는 비용도 매우 커지게 되었다._

In addition, inaccurate imputation might distort the data considerably.
_추가로 부정확한 imputation은 오히려 데이터를 왜곡하여 성능을 크게 저하시켰다._

Hence, more recent works suggested modeling directly the observed ratings only, while avoiding overfitting through a regularized model. 
_따라서 최근의 연구는 평점 데이터가 있는 것만 다루되, 규제화를 통하여 overfitting을 피하는 방법을 고안하였다._

To learn the factor vectors ($p_u$ and $q_i$), the system minimizes the regularized squared error on the set of known ratings:
_잠재 벡터를 학습하기 위해서, 추천시스템은 존재하는 평점 데이터에 대해서만 규제화 term이 추가된 예측 오차제곱합을 줄이도록 학습을 한다._
$$\tag{2} \min_{q*, p*} \sum_{(u, i) \in \mathcal{K}} (r_{ui}-q_i^\top p_u)^2 + \lambda (\| q_i\|^2 + \|p_u\|^2)$$

Here, $\mathcal{K}$ is the set of the ($u,i$) pairs for which $r_{ui}$ is known (the training set).
_여기서 $\mathcal{K}$ 가 의미하는 것은, 실제 존재하는 평점에 대한 (유저, 아이템)쌍을 말한다. (학습데이터셋 에서)_

The system learns the model by fitting the previously observed ratings.
_추천시스템은 이 $\mathcal{K}$ 를 통해서 학습을 진행한다._

However, the goal is to generalize those previous ratings in a way that predicts future, unknown ratings.
_하지만 추천시스템의 목적은 보지 않은 평점에 대해서도 예측을 잘 하는 일반화된 예측을 잘해야 하는 것이다._

Thus, the system should avoid overfitting the observed data by regularizing the learned parameters, whose magnitudes are penalized.
_따라서, 추천시스템은 이미 존재하는 평점에 대해 overfitting이 되어서는 안되기 때문에 학습 파라미터에 규제화를 적용하였다._

The constant $\lambda$ controls the extent of regularization and is usually determined by cross-validation.
_상수 $\lambda$는 규제화의 정도를 조절하는 파라미터이며, 값은 cross-validation을 통해서 결정된다._

Ruslan Salakhutdinov and Andriy Mnih’s “Probabilistic Matrix Factorization” offers a probabilistic foundation for regularization.
_Ruslan Salakhutdinov와 Andriy Mnih의 "Porbabilistic Matrix Factorization"은 규제화에 확률적 개념을 적용시켰다._

<br>

# Learning algirthms

Two approaches to minimizing Equation 2 are stochastic gradient descent and alternating least squares (ALS).
_바로 위의 규제화가 포함된 Matrix Factorization 수식을 최적화 시키는 방법에는 '경사하강법'과 'alternating least squares(ALS)' 2가지 방법이 있다._

<br>

## Stochastic gradient descent

Simon Funk popularized a stochastic gradient descent optimization of Equation 2 wherein the algorithm loops through all ratings in the training set.
_Simon Funk는 위의 수식에 대해 모든 학습 데이터셋에 존재하는 평점들을 순회하며 SGD를 적용하는 방법을 발표하였다._

For each given training case, the system predicts $r_{ui}$ and computes the associated prediction error
*주어진 학습 데이터마다 추천시스템은 평점 $r_{ui}$ 를 예측하고 예측오차를 계산한다.*

$$e_{ui} \overset{def}{=} r_{ui} - q_i^\top p_u$$

Then it modifies the parameters by a magnitude proportional to $\gamma$ in the opposite direction of the gradient, yielding:
_그런 다음 파라미터를 $\gamma$ 만큼 gradient의 반대 방향으로 업데이트(수정)한다._

- $q_i \leftarrow q_i + \gamma \cdot (e_{ui}\cdot p_u - \lambda \cdot q_i)$
- $p_u \leftarrow p_u + \gamma \cdot (e_{ui}\cdot p_i - \lambda \cdot p_u)$


This popular approach combines implementation ease with a relatively fast running time.
_이러한 접근 방식은 비교적 구현하기 쉬우며 빠르게 작동한다._

Yet, in some cases, it is beneficial to use ALS optimization.
_하지만 몇 가지 경우에는 ALS 최적화를 사용하는 것이 더 이득이다._

<br>

## Alternating least squares

Because both $q_i$ and $p_u$ are unknowns, Equation 2 is not convex.
*$q_i$와 $p_u$는 모르는 값들이기 때문에 Equation2가 convex가 아닐 수 있다.*

However, if we fix one of the unknowns, the optimization problem becomes quadratic and can be solved optimally. 
*하지만, 만약 모르는 값들 중 하나를 고정시킨다면, 최적화 문제는 quadratic 형태가 되며 최적의 해를 찾는 것이 가능하다.*

Thus, ALS techniques rotate between fixing the $q_i$’s and fixing the $p_u$’s. 
*따라서, ALS는 번갈아가면서 $q_i$와 $p_u$를 고정시켜서 최적화한다.*

When all $p_u$’s are fixed, the system recomputes the $q_i$’s by solving a least-squares problem, and vice versa. 
*만약 모든 $p_u$가 고정되어 있다면, 추천시스템은 $q_i$를 least-square 문제를 푸는 것으로 계산한다. (반대의 경우도 마찬가지)*

This ensures that each step decreases Equation 2 until convergence.
*이 방법은 각 step마다 Equation2가 수렴할 때 까지 감소시킴을 보장한다.*

While in general stochastic gradient descent is easier and faster than ALS, ALS is favorable in at least two cases. 
*일반적인 SGD가 ALS보다 빠르지만, ALS은 다음의 2가지 경우에서는 더 선호되는 최적화 방법이다.*

The first is when the system can use parallelization.
*첫 번째 경우는 시스템이 병렬 처리가 가능한 경우이다.*

In ALS, the system computes each $q_i$ independently of the other item factors and computes each $p_u$ independently of the other user factors. 
*ALS에서는 추천시스템이 다른 아이템 요소들을 보고 $q_i$를 계산하고, 다른 유저 요소들을 보고 $p_u$를 각각 독립적으로 계산한다.*

This gives rise to potentially massive parallelization of the algorithm.
*이것은 잠재적으로 알고리즘이 대규모로 병렬화 될 수 있음을 말한다.*

The second case is for systems centered on implicit data. 
*두 번째는 implicit data가 많은 경우이다.*

Because the training set cannot be considered sparse, looping over each single training case—as gradient descent does—would not be practical. 
*implicit data가 많은 경우에는 유저-아이템 상호작용 행렬이 희소(sparse)하지 않기 때문에, 존재하는 모든 rating을 돌면서 최적화 하는 것은 실용적이지 못하다.*

ALS can efficiently handle such cases.
*이런 경우 ALS를 사용하면 효과적으로 문제를 해결할 수 있다.*

<br>

# Adding biases

One benefit of the matrix factorization approach to collaborative filtering is its flexibility in dealing with various data aspects and other application-specific requirements. 
*Matrix factorization을 사용한 협업필터링 접근법은 다양한 데이터 양상을 띄거나 특정 도메인의 데이터를 사용할 때 유연하다는 장점이 있다.*

This requires accommodations to Equation 1 while staying within the same learning framework. 
*Matrix factorization은 Equation1을 계속 유지하면서 학습을 시킬 수 있다.*

Equation 1 tries to capture the interactions between users and items that produce the different rating values. 
*Equation 1은 유저와 아이템 사이의 관계를 포착하여 서로 다른 평점 값을 예측하도록 만든다.*

However, much of the observed variation in rating values is due to effects associated with either users or items, known as biases or intercepts, independent of any interactions.
*하지만 평점 값의 넓은 변동 값은 사용자나 아이템에 존재하는 bias와 관련된 효과로 발생한다.*

For example, typical collaborative filtering data exhibits large systematic tendencies for some users to give higher ratings than others, and for some items to receive higher ratings than others. 
*예를 들어, 전형적인 협업필터링 데이터의 데이터는 몇 개의 특징들이 존재한다. 그 종류는 다른 사람들보다 높은 평점을 주는 사람이 있을 수 있고, 몇 개의 아이템들을 이 때문에 더 높은 평점을 받고 있는 것들이다.*

After all, some products are widely perceived as better (or worse) than others.
*결국, 몇 개의 아이템(상품)들은 다른 아이템들보다 더 좋은 평가 또는 더 안좋은 평가를 받게 된다.*

Thus, it would be unwise to explain the full rating value by an interaction of the form $q_i^\top p_u$.
*따라서, Matrix factorization의 Equation 1인 $q_i^\top p_u$ 만으로 상호작용을 설명하는 것은 현명하지 못하다.*

Instead, the system tries to identify the portion of these values that individual user or item biases can explain, subjecting only the true interaction portion of the data to factor modeling. 
*대신, 추천시스템은 이러한 유저나 아이템에 대한 bias를 설명할 수 있는 부분에 대해서만 식별하려고 하며, 데이터의 실제 상호작용이 있는 부분만 모델링에 적용한다.*

A first-order approximation of the bias involved in rating $r_{ui}$ is as follows:
*평점 $r_{ui}$를 예측하는데 포함될 bias값의 first-order approximation은 다음과 같다:*

$$\tag{3} b_{ui} = \mu + b_i + b_u$$

The bias involved in rating $r_{ui}$ is denoted by $b_{ui}$ and accounts for the user and item effects. 
*bias $b_{ui}$ 가 포함된 평점 $r_{ui}$ 는 유저나 아이템이 불러일으키는 bias요소들을 다룰 수 있다.*

The overall average rating is denoted by $\mu$; the parameters $b_u$ and $b_i$ indicate the observed deviations of user $u$ and item $i$, respectively, from the average.
*전체 평점에 대한 평균은 $\mu$ 라고 나타낸다; 파라미터 $b_u$와 $b_i$ 는 유저와 아이템 각각에 대해 계산측된 편차를 말한다.*

For example, suppose that you want a first-order estimate for user Joe’s rating of the movie Titanic.
*예를 들어, 우리가 영화 Titanic에 대해 Joe가 내릴 평점을 예측한다고 가정해보자.*

Now, say that the average rating over all movies, $\mu$, is 3.7 stars.
*모든 영화에 대한 평균 평점 $\mu$가 3.7 이라고 하자.*

Furthermore, Titanic is better than an average movie, so it tends to be rated 0.5 stars above the average. 
*또한, Titanic은 다른 영화들의 평균 평점보다 더 높아서 평균보다 0.5점 이상 평점이 매겨지고 있다. (가정)*

On the other hand, Joe is a critical user, who tends to rate 0.3 stars lower than the average.
*반면, Joe는 상당히 비판적인 유저이기 때문에 대부분의 평점을 평균보다 0.3정도 이하로 매긴다.*

Thus, the estimate for Titanic’s rating by Joe would be 3.9 stars (3.7 + 0.5 - 0.3). 
*따라서, Joe가 Titanic 영화에 내릴 평점은 3.7 + 0.5 - 0.3 으로 예측할 수 있게 된다.*

Biases extend Equation 1 as follows:
*Equation 1에 bias가 추가된 형태는 다음과 같다.*

$$\tag{4} \hat{r}_{ui} = \mu + b_i + b_u + q_i^\top p_u$$

Here, the observed rating is broken down into its four components: global average, item bias, user bias, and useritem interaction.
*이제 평점은 4개의 컴포넌트로 분해될 수가 있다: 평균 평점, 아이템 bias, 유저 bias, 유저-아이템 상호작용*

This allows each component to explain only the part of a signal relevant to it.
*이를 통해 각 컴포넌트는 관련된 값들(bias term)에 대해서만 설명할 수 있게 된다.*

The system learns by minimizing the squared error function:
*추천시스템은 위의 수식을 오차 제곱을 최소화하는 방향으로 학습한다.*

$$\tag{5} \min_{p*, q*, b*} \sum_{(u, i)\in \mathcal{K}} (r_{ui} - \mu - b_u - b_i - p_u^\top q_i)^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)$$

Since biases tend to capture much of the observed signal, their accurate modeling is vital.
*편향은 관측된 신호의 대부분을 포착하는 경향이 있기 때문에 정확한 모델링이 필수적이다.*

Hence, other works offer more elaborate bias models.
*따라서 다른 연구들은 더 정교한 bias 모델을 제공한다.*

<br>

# Additional input sources

Often a system must deal with the cold start problem, wherein many users supply very few ratings, making it difficult to reach general conclusions on their taste.
*종종 추천시스템은 cold start 문제를 다루어야 하며, 이 문제는 유저들이 적은 평점만을 남겨서 그들의 선호도를 일반화하기 어려운 것을 말한다.*

A way to relieve this problem is to incorporate additional sources of information about the users. 
*이 문제를 해결하는 방법 중 하나는 그들의 정보에 대한 추가 정보를 학습에 사용하는 것이다.*

Recommender systems can use implicit feedback to gain insight into user preferences. 
*추천시스템은 implicit feedback을 사용하여 그들의 선호도를 파악하는 것이 가능하다.*

Indeed, they can gather behavioral information regardless of the user’s willingness to provide explicit ratings. 
*실제로, 유저가 평점을 몇 점으로 남겼는지의 명시적인 데이터와 상관없이 행동 정보를 수집할 수 있다.*

A retailer can use its customers’ purchases or browsing history to learn their tendencies, in addition to the ratings those customers might supply. 
*판매자들(서비스 제공 업체)은 고객의 구매 또는 검색 이력을 통해 선호도를 파악할 수 있고, 이는 이후 고객의 평점 예측을 하는데 사용될 수 있다.*

For simplicity, consider a case with a Boolean implicit feedback. 
*단순화를 위해 implicit feedback 중에서도 Boolean 형태를 고려해보도록 하자.*

$N(u)$ denotes the set of items for which user $u$ expressed an implicit preference. 
*$N(u)$는 $u$가 implicit 선호도를 표현한 아이템들의 집합을 말한다.*

This way, the system profiles users through the items they implicitly preferred. 
*여기서 추천시스템은 유저의 선호도를 implicit하게 선호한 아이템들의 집합 $N(u)$를 사용하여 모델링한다.*

Here, a new set of item factors are necessary, where item $i$ is associated with $x_i \in \mathbb{R}^f$ .
*여기서 새로운 아이템의 잠재 요소들이 필요하게 되며, 아이템 $i$는 잠재벡터로 $x_i$로 나타내진다.*

Accordingly, a user who showed a preference for items in $N(u)$ is characterized by the vector $\sum_{i \in N(u)} x_i$
*를 선호하였다면, 그의 선호도는 아이템 벡터들을 사용하여 계산된다: $\sum_{i \in N(u)} x_i$*

Normalizing the sum is often beneficial, for example, working with $\vert N(u) \vert^{-0.5} \sum_{i\in N(u)}x_i$.
*합을 정규화하는 것은 종종 도움이 되기 때문에 $\vert N(u) \vert^{-0.5} \sum_{i\in N(u)}x_i$ 처럼 표현할 수 있다.*

Another information source is known user attributes, for example, demographics.
*또 따른 추가 정보로 인구통계학 정보와 같은 유저의 속성을 사용할 수 있다.*

Again, for simplicity consider Boolean attributes where user $u$ corresponds to the set of attributes $A(u)$, which can describe gender, age group, Zip code, income level, and so on. 
*다시 단순화를 위해 Boolean 속성만 고려한다하면, 유저 $u$에 상응하면 속성의 집합 $A(u)$는 성별, 나이대, 우편번호, 수입레벨 등을 나타낼 수 있다.*

A distinct factor vector $y_a \in \mathbb{R}^f$ corresponds to each attribute to describe a user through the set of user-associated attributes: $\sum_{a \in A(u)} y_a$.
*마찬가지로 유저의 선호도를 계산하기 위해 유저와 관련된 속성들의 합으로 나타내는 것도 가능하다.*

The matrix factorization model should integrate all signal sources, with enhanced user representation:
*Matrix factorization 모델은 이제 위에서 살펴본 모든 컴포넌트들을 하나로 합쳐서 유저 잠재벡터의 표현을 생성한다.*

$$\tag{6} \hat{r}_{ui} = \mu + b_i + b_u + q_i^\top [p_u + \vert N(u) \vert^{-0.5} \sum_{i\in N(u)}x_i + \sum_{a \in A(u)} y_a]$$

While the previous examples deal with enhancing user representation—where lack of data is more common— items can get a similar treatment when necessary.
*부족한 데이터를 보완하기 위해서 현재는 유저의 잠재 벡터 표현력을 강화하는데에 예시를 들었지만, 아이템의 정보가 충분하다면 유사한 방법으로 아이템 잠재벡터도 개선가능하다.*

<br>

# Temporal dynamics

So far, the presented models have been static.
*지금까지 소개한 모델은 정적인 부분만 다루었다.*

In reality, product perception and popularity constantly change as new selections emerge.
*사실, 상품의 인식이나 인기도는 새로운 상품이 등장할 때마다 변화한다.*

Similarly, customers’ inclinations evolve, leading them to redefine their taste. 
*유사하게, 고객의 선호도 또한 변화하며 이전과 다른 상품을 좋아할 수도 있게 된다.*

Thus, the system should account for the temporal effects reflecting the dynamic, time-drifting nature of user-item interactions. 
*따라서 추천시스템은 변화하는 시간속에서 유저-아이템 상호작용의 변화를 포착할 수 있어야 한다.*

The matrix factorization approach lends itself well to modeling temporal effects, which can significantly improve accuracy. 
*matrix factorization 접근은 이러한 시간적 요소들을 고려하여 모델링을 할 수 있어 정확도를 크게 개선시킬 수 있다.*

Decomposing ratings into distinct terms allows the system to treat different temporal aspects separately.
*평점을 서로 다른 term으로 분해하는 것은 추천시스템이 각 요소들에게 서로 다른 시간적 요소를 고려할 수 있게 만들어준다.*

Specifically, the following terms vary over time: item biases, bi (t); user biases, bu(t); and user preferences, pu(t). 
*특히 다음 term들은 시간에 따라 크게 변화한다: 아이템 bias $b_i(t)$,  유저 bias $b_u(t)$, 유저 선호도 $p_u(t)$ *

The first temporal effect addresses the fact that an item’s popularity might change over time.
*첫 번째 시간적 요소를 다루는 아이템 bias는 시간이 지남에 따라 변화하는 아이템의 인기도를 다룬다.*

For example, movies can go in and out of popularity as triggered by external events such as an actor’s appearance in a new movie.
*예를 들어, 영화의 인기도는 새로운 영화에 등장하는 배우의 출연에 따라 내려가거나 올라올 수 있다.*

Therefore, these models treat the item bias $b_i$ as a function of time.
*따라서, 모델은 아이템에 대한 bias를 시간에 따라 변화하도록 모델링하여야 한다.*

The second temporal effect allows users to change their baseline ratings over time. 
*두 번째로 시간적 요소를 다루는 것은 시간이 지남에 따라 변화하는 유저의 기준 평점이다.*

For example, a user who tended to rate an average movie “4 stars” might now rate such a movie “3 stars.” 
*예를 들어, 평균 평점으로 4점을 주던 사람은 시간이 지나 이제는 3점을 줄 수도 있다.*

This might reflect several factors including a natural drift in a user’s rating scale, the fact that users assign ratings relative to other recent ratings, and the fact that the rater’s identity within a household can change over time.
*이것은 유저가 평점을 내리는 것에 대한 자연스러운 변화, 최근 평점과 관련해서 평점을 부여하는 경향 등 자연스럽게 변화하기 때문일 수 있다.*

Hence, in these models, the parameter $b_u$ is a function of time.
*따라서 모델에서는 파라미터 $b_u$를 시간에 따라 변화하도록 모델링하였다.*

Temporal dynamics go beyond this; they also affect user preferences and therefore the interaction between users and items.
*변화하는 시간은 유저의 선호도에 영향을 미칠 수 있고, 유저와 아이템의 상호작용에도 영향을 줄 수 있다.*

Users change their preferences over time.
*유저는 오랜 기간동안 지속해서 선호도가 변화한다.*

For example, a fan of the psychological thrillers genre might become a fan of crime dramas a year later.
*예를 들어, 심리 스릴러 장르를 좋아하던 사람은 몇 년후에 범죄 드라마를 좋아할 수 있다.*

Similarly, humans change their perception of certain actors and directors.
*비슷하게, 사람들이 배우나 감독에게 갖는 인지도 역시 변화한다.*

The model accounts for this effect by taking the user factors (the vector $p_u$) as a function of time.
*모델은 이런 변화를 유저의 잠재요소 $p_u$를 시간에 따라 변화하게 하여 모델링했다.*

On the other hand, it specifies static item characteristics, $q_i$ , because, unlike humans, items are static in nature.
*반면, 아이템 잠재벡터 즉, 아이템의 특징들은 사람과 다르게 그 상태 그대로 유지한다.*

Exact parameterizations of time-varying parameters lead to replacing Equation 4 with the dynamic prediction rule for a rating at time $t$:
*변화하는 시간을 Matrix factorization 수식에 포함시킨 결과는 다음과 같다:*

$$\tag{7} \hat{r}_{ui}(t) = \mu + b_i(t) + b_u(t) + q_i^\top p_u(t)$$

<br>

# Inputs with varying confidence levels

In several setups, not all observed ratings deserve the same weight or confidence. 
*관측된 모든 평점들이 동일한 가중치나 신뢰도를 갖는 것은 아니다.*

For example, massive advertising might influence votes for certain items, which do not aptly reflect longer-term characteristics. 
*예를 들어, 광고를 위해 엄청 난 수의 평점들이 조작되었다면, 이는 평점에 큰 영향을 주어서는 안된다.*

Similarly, a system might face adversarial users that try to tilt the ratings of certain items. 
*유사하게, 추천시스템은 이런 광고성 유저들을 마주하게 될 수도 있기 때문에, 광고 평점에 대한 문제를 해결해야 한다.*

Another example is systems built around implicit feedback. 
*또 다른 예시는 implicit feedback으로 만들어진 추천시스템이다.*

In such systems, which interpret ongoing user behavior, a user’s exact preference level is hard to quantify. 
*이런 추천시스템에서는 유저의 행동이 정확히 해당 아이템을 선호하는지 알 수 없기 때문에 문제가 될 수 있다.*

Thus, the system works with a cruder binary representation, stating either “probably likes the product” or “probably not interested in the product.” 
*따라서 추천시스템은 "아마도 제품을 좋아할 것이다" 또는 "아마도 제품에 관심이 없을 것이다"와 같이 대충 측정된 데이터들을 다뤄야 한다.*

In such cases, it is valuable to attach confidence scores with the estimated preferences. 
*이러한 경우에는, 예측한 평점에 신뢰도를 함께 제공하여 주는것이 유용하다.*

Confidence can stem from available numerical values that describe the frequency of actions, for example, how much time the user watched a certain show or how frequently a user bought a certain item. 
*신뢰도는 유저가 특정  TV쇼 또는 특정 아이템을 구입한 횟수와 같은 빈도를 사용하는 숫자값 같은 것을 사용할 수 있다.*

These numerical values indicate the confidence in each observation.
*이러한 수치적 값들은 각 관측된 데이터의 신뢰도를 나타낸다.*

Various factors that have nothing to do with user preferences might cause a one-time event; however, a recurring event is more likely to reflect user opinion. 
*유저 선호도와 관련이 없는 여러 요인들은 일회성 이벤트를 발생시키지만, 반복되는 이벤트는 사용자의 선호도를 반영할 가능성이 높다.*

The matrix factorization model can readily accept varying confidence levels, which let it give less weight to less meaningful observations. 
*matrix factoriztion 모델은 다양한 신뢰 수준을 쉽게 수용할 수 있으므로, 덜 중요한 평점 값에 대해서는 가중치를 덜 부여할 수 있다.*

If confidence in observing $r_{ui}$ is denoted as $c_{ui}$, then the model enhances the cost function (Equation 5) to account for confidence as follows:
*만약 어떤 평점 $r_{ui}$에 대한 신뢰도를 $c_{ui}$라고 한다면 confidence를 포함한 추천시스템 모델의 손실함수는 다음과 같다:*

$$\tag{8} \min_{p*, q*, b*}\sum_{(u, i) \in \mathcal{K}} c_{ui} (r_{ui} - \mu - b_i -b_u -p_u^\top q_i) + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)$$

For information on a real-life application involving such schemes, refer to "Collaborative Filtering for Implicit Feedback Datasets."
*이런 내용을 더 자세하게 담고있고, 실생활에 적용가능한 implicit feedback데이터를 사용한 추천시스템에 대한 내용은 "Collaborative Filtering for Implicit Feedback Datasets."을 참고해라.*

