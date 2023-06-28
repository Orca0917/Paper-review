

As the Netflix Prize competition has demonstrated, matrix factorization models are superior to classic nearest-neighbor techniques for producing product recommendations, allowing the incorporation of additional information such as implicit feedback, temporal effects, and confidence levels.
_Netflix Prize 경진대회에서 입증했듯이, Matrix Factorization 모델은 implicit feedback이나 시간적 요소, 신뢰도 레벨과 같은 추가정보를 사용하여 이전 아이템 추천을 해주는 Nearest-neighbor 기법보다 더 우수한 성적을 보여주었다._

Modern consumers are inundated with choices.
_현대의 소비자들은 선택의 늪에 빠져있다._

Electronic retailers and content providers offer a huge selection of products, with unprecedented opportunities to meet a variety of special needs and tastes.
_전자상거래 판매자들과 컨텐츠 제공자들은 소비자들이 다양한 욕구와 입맛에 맞게 상품을 선택할 수 있게 하고 있다._

Matching consumers with the most appropriate products is key to enhancing user satisfaction and loyalty.
_판매자들은 소비자들이 원하는 상품들을 매칭시켜줌으로써 사용자들의 만족감과 충성심을 높일 수 있다._

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


# A basic matrix factorization model