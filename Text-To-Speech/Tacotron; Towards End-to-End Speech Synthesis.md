
# Abstact

A text-to-speech synthesis system typically consists of multiple stages, such as a text analysis frontend, an acoustic model and an audio synthesis module.
TTS 음성합성 시스템은 일반적으로 여러 단계로 구성되어 있다. (텍스트 분석 frontend + acoustic model + 음성합성 모듈)

Building these components often requires extensive domain expertise and may contain brittle design choices.
이런 구성요소들을 사용하는 것은 광범위한 음성 도메인 전문지식이 필요할 수 있고, 불안정한 구조를 갖게 된다.

In this paper, we present Tacotron, an end-to-end generative text-to-speech model that synthesizes speech directly from characters.
앞선 구조와는 달리, 본 논문에서는 글자로 부터 음성을 바로 학습시키는 end-to-end TTS 생성 모델인 Tacotron을 소개한다.

Given <text, audio> pairs, the model can be trained completely from scratch with random initialization.
Tacotron은 <텍스트, 오디오>쌍이 주어지면 모델은 임의의 초기화된 상태인 밑바닥부터 학습을 시작한다.

We present several key techniques to make the sequence-to-sequence framework perform well for this challenging task.
우리는 어려운 과제를 잘 해결할 수 있는 sequence-to-sequence 프레임워크를 만드는 몇가지 주요 기술을 소개한다.

Tacotron achieves a 3.82 subjective 5-scale mean opinion score on US English, outperforming a production parametric system in terms of naturalness.
결과를 보자면, Tacotron은 US 영어에 대해 5점 만점 중 3.82라는 객관적인 평균점수를 보여주었고, 이는 음성의 자연스러움 측면에서 이전의 parametric 시스템 성능을 능가한다.

In addition, since Tacotron generates speech at the frame level, it’s substantially faster than sample-level autoregressive methods.
추가로, Tacotron은 음성을 frame 단위로 생성하기 때문에, sample단위의 autoregressive 방법보다 훨씬 빠르게 동작한다.

# 1. Introduction

Modern text-to-speech (TTS) pipelines are complex (Taylor, 2009).
“현대의 TTS 파이프라인은 복잡하다” — (Taylor, 2009).

For example, it is common for statistical parametric TTS to have a text frontend extracting various linguistic features, a duration model, an acoustic feature prediction model and a complex signal-processing-based vocoder (Zen et al., 2009; Agiomyrgiannakis, 2015).
예를 들어, ==Statistical Parametric TTS==는 일반적으로 다양한 언어학적 특징을 추출하는 text frontend$^1$와 duration 모델$^2$, acoustic feature 예측 모델$^3$ 그리고 복잡한 신호처리 기반 vocoder$^4$를 필요로 하였다.

These components are based on extensive domain expertise and are laborious to design.
위 요소들은 광범위한 전문적인 도메인 지식을 기반으로 하였고 모델 구조를 설계하는데 힘이 많이 들었다.

They are also trained independently, so errors from each component may compound.
또한, 위 구성요소들은 각각 독립적으로 학습되기에, 각 컴포넌트들의 에러들은 모두 합쳐져서 더 큰 에러를 발생시키게 되었다.

The complexity of modern TTS designs thus leads to substantial engineering efforts when building a new system.
따라서 현대의 복잡한 TTS 설계는 새로운 시스템을 구축하는데 있어서 엄청난 엔지니어링 노력을 필요로 하였다.

There are thus many advantages of an integrated end-to-end TTS system that can be trained on <text, audio> pairs with minimal human annotation.
따라서, <텍스트, 오디오>쌍을 기반으로 학습되고 사람의 노력을 최소로 하는 하나로 통합된 end-to-end TTS 시스템의 장점에는 여러가지가 존재한다.

First, such a system alleviates the need for laborious feature engineering, which may involve heuristics and brittle design choices.
첫째, 휴리스틱하고 불안정한 설계를 수반하는 힘든 feature engineering 단계를 end-to-end TTS 시스템이 완화시킬 수 있다.

Second, it more easily allows for rich conditioning on various attributes, such as speaker or language, or high-level features like sentiment.
둘째, 화자 또는 사용하는 언어와 같은 다양한 속성이나 감정과 같은 고급 속성들에 대한 학습을 허용한다.

This is because conditioning can occur at the very beginning of the model rather than only on certain components.
이것은 학습하는 과정이 각 구성요소에서 진행하지 않고, 모델의 가장 처음부분에서만 진행되기 때문이다.

Similarly, adaptation to new data might also be easier.
비슷하게, 새로운 데이터에 대한 적용도 더 쉽게 진행된다.

Finally, a single model is likely to be more robust than a multi-stage model where each component’s errors can compound.
마지막으로, 하나의 모델을 사용하는 것이 여러 개의 모델을 사용하여 에러들이 합쳐지는 것보다 robust하다.

These advantages imply that an end-to-end model could allow us to train on huge amounts of rich, expressive yet often noisy data found in the real world.
이러한 장점들로 보았을 때, end-to-end 모델을 통해 실생활에서 찾아볼 수 있는 표현력(속성)이 풍부한 소음이 많은 데이터에서도 학습을 진행할 수 있다는 것을 내포한다.

TTS is a large-scale inverse problem: a highly compressed source (text) is “decompressed” into audio.
TTS는 대규모 역문제(Inverse problem)이다: 많이 압축(compressed)된 source(텍스트)가 오디오로 압축해제(decompressed)된다.

Since the same text can correspond to different pronunciations or speaking styles, this is a particularly difficult learning task for an end-to-end model: it must cope with large variations at the signal level for a given input.
동일한 텍스트 이더라도 억양이나 발화 스타일에 따라서 다른 음성으로 대응될 수 있기 때문에, end-to-end 모델에서는 특히 어려운 학습으로 간주된다: 모델은 반드시 넓은 분포(변화량)의 신호 입력에 대해서도 대응할 수 있어야 한다.

Moreover, unlike end-to-end speech recognition (Chan et al., 2016) or machine translation (Wu et al., 2016), TTS outputs are continuous, and output sequences are usually much longer than those of the input.
게다가, end-to-end 음성 인식이나 기계 번역과는 다르게 TTS는 연속적인 출력을 내보내며 그 길이도 입력보다 훨씬 길다.

These attributes cause prediction errors to accumulate quickly.
이런 특징들이 예측 오류를 빠르게 증가시키는 원인이 된다.

In this paper, we propose Tacotron, an end-to-end generative TTS model based on the sequence-to-sequence (seq2seq) (Sutskever et al., 2014) with attention paradigm (Bahdanau et al., 2014).
본 논문에서 우리는 Tacotron이라는 모델을 발표하였으며, 이 모델은 attention 패러다임을 사용한 sequence-to-sequence에 기반을 둔 end-to-end TTS 생성모델이다.

Our model takes characters as input and outputs raw spectrogram, using several techniques to improve the capability of a vanilla seq2seq model.
우리의 모델은 문자를 입력으로 받고 출력으로 spectrogram을 생성하며, 몇 가지 테크닉을 통해 Tacotron에 사용되는 기본 seq2seq 모델의 성능을 증진시켰다.

Given <text, audio> pairs, Tacotron can be trained completely from scratch with random initialization.
<텍스트, 오디오>쌍이 주어지면 모델은 임의의 초기화된 상태인 밑바닥부터 학습을 시작한다.

It does not require phoneme-level alignment, so it can easily scale to using large amounts of acoustic data with transcripts.
Tacotron모델은 음소 단위의 alignment가 필요하지 않으므로 대본(transcripts)이 포함된 대용량의 acoustic data로 확장하는 것이 용이하다.

With a simple waveform synthesis technique, Tacotron produces a 3.82 mean opinion score (MOS) on an US English eval set, outperforming a production parametric system in terms of naturalness.
추가로 간단한 waveform 합성 기술을 사용하여 Tacotron은 영어 평가데이터셋에 대해 3.82 MOS점수를 받았으며 이는 자연스러움 측면에서 기존의 parametric 시스템을 능가하는 수치이다.

# 2. Related Work

WaveNet (van den Oord et al., 2016) is a powerful generative model of audio.

WaveNet은 오디오를 생성하는 강력한 모델이다.

It works well for TTS, but is slow due to its sample-level autoregressive nature.

WaveNet은 TTS에서 잘 작동하지만, sample단위의 autoregressive 구조를 사용하기 때문에 학습속도가 느리다.

It also requires conditioning on linguistic features from an existing TTS frontend, and thus is not end-to-end: it only replaces the vocoder and acoustic model.

또한 WaveNet은 존재하는 TTS frontend에서 언어학적 특징에 대해 별도로 학습하는 과정을 필요로 하기 때문에 end-to-end라고 볼 수 없다: WaveNet은 오직 vocoder와 acoustic model만 대체한다.

Another recently-developed neural model is DeepVoice (Arik et al., 2017), which replaces every component in a typical TTS pipeline by a corresponding neural network.

최근에 개발된 또 다른 신경망 모델은 DeepVoice이며 이것은 각 컴포넌트들의 전형적인 TTS 파이프라인을 신경망 구조로 대체하였다.

However, each component is independently trained, and it’s nontrivial to change the system to train in an end-to-end fashion.

하지만, 각 컴포넌트들은 별도로 학습이 되어야하며, DeepVoice는 end-to-end 방식으로 학습하게 시스템을 변경하는 것에는 기여하지 않았다.

To our knowledge, Wang et al. (2016) is the earliest work touching end-to-end TTS using seq2seq with attention.

우리의 지식에 따르면, Wang이 2016년도에 발표한 논문이 attention을 사용한 seq2seq를 통해 end-to-end TTS를 만들려고 한 최초의 시도이다.

However, it requires a pre-trained hidden Markov model (HMM) aligner to help the seq2seq model learn the alignment.

하지만, 해당 논문에서 발표한 구조는 seq2seq가 alignment를 학습하기 위해 기학습된 Hidden Markov Model (HMM) aligner를 필요로 하였다.

It’s hard to tell how much alignment is learned by the seq2seq per se.

이것은 seq2seq 자체만으로 alignment가 얼마나 학습되었는지 알 수 없게 만든다.

Second, a few tricks are used to get the model trained, which the authors note hurts prosody.

둘째, 모델을 훈련시키기 위해 몇가지 트릭이 사용되었는데, 저자들은 이것이 prosody(운율)에 있어서 해를 끼친다고 보았다.

Third, it predicts vocoder parameters hence needs a vocoder.

셋째, vocoder를 필요로하기 대문에 vocoder parameter를 예측해야한다.

Furthermore, the model is trained on phoneme inputs and the experimental results seem to be somewhat limited.

게다가, 모델은 음소단위의 입력을 받기 때문에 실험 결과가 다소 제한적일 것으로 보인다.

Char2Wav (Sotelo et al., 2017) is an independently-developed end-to-end model that can be trained on characters.

2017년에 발표된 Char2Wav는 문자로 학습될 수 있는 독립적으로 개발된 end-to-end 모델이다.

However, Char2Wav still predicts vocoder parameters before using a SampleRNN neural vocoder (Mehri et al., 2016), whereas Tacotron directly predicts raw spectrogram.

하지만 Char2Wav는 여전히 SampleRNN 신경망 vocoder를 통해 vocoder parameter를 예측해야 했던 반면, Tacotron은 바로 spectrogram을 예측한다.

Also, their seq2seq and SampleRNN models need to be separately pre-trained, but our model can be trained from scratch.

또한, Char2Wav의 seq2seq와 SampleRNN 모델은 별도로 기학습되어야 했지만, 우리의 모델(Tacotron)은 밑바닥부터 훈련할 수 있다.

Finally, we made several key modifications to the vanilla seq2seq paradigm.

마지막으로, 우리는 기본(vanilla) seq2seq 패러다임에 몇 가지 주요 변화를 주었다.

As shown later, a vanilla seq2seq model does not work well for character-level inputs.

나중에 보이겠지만, 기본 seq2seq 모델은 문자단위의 입력에 대해 좋은 성능을 보이지 못한다.

# 3. Model Architecture

The backbone of Tacotron is a seq2seq model with attention (Bahdanau et al., 2014; Vinyals et al., 2015).

Tacotron 모델의 중추는 attention을 사용한 seq2seq 모델이다.

Figure 1 depicts the model, which includes an encoder, an attention-based decoder, and a post-processing net.

Figure 1은 모델의 구조를 묘사하며, encoder, attention기반 decoder, 후처리 신경망으로 구성되어 있다.

At a high-level, our model takes characters as input and produces spectrogram frames, which are then converted to waveforms.

높은 수준에서 바라보면, 우리의 모델은 문자를 입력으로 받아 spectrogram frame을 생성해낸다. 이 spectrogram은 이 후 waveform으로 변환된다.

We describe these components below.

우리는 이 구성요소들을 아래에서 상세하게 설명한다.

![Figure 1: 모델 구조. 모델은 문자를 입력으로 받아 출력으로 입력에 상응하는 spectrogram을 생성하였고, spectrogram은 이후 Griffin-Lim 알고리즘을 사용하여 음성으로 변환된다.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3c56d324-5eab-44ed-a233-53d2b1c172e7/Untitled.png)

Figure 1: 모델 구조. 모델은 문자를 입력으로 받아 출력으로 입력에 상응하는 spectrogram을 생성하였고, spectrogram은 이후 Griffin-Lim 알고리즘을 사용하여 음성으로 변환된다.

### 3.1 CBHG Module

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f31491c6-ba36-4e82-b8be-b8aec4092d9a/Untitled.png)

We first describe a building block dubbed CBHG, illustrated in Figure 2.

우리는 CBHG라고 부르는 building block을 Figure 2에 묘사했다.

CBHG consists of a bank of 1-D convolutional filters, followed by highway networks (Srivastava et al., 2015) and a bidirectional gated recurrent unit (GRU) (Chung et al., 2014) recurrent neural net (RNN).

CBHG는 1D Convolution 필터로 쌓아올려져 있고, 이후에는 highway network와 양방향 GRU RNN이 존재한다.

CBHG is a powerful module for extracting representations from sequences.

CBHG는 sequence로 부터 표현(representation)을 추출하는 강력한 모듈이다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/91f3af94-822b-4397-b88d-b320ea744268/Untitled.png)

The input sequence is first convolved with $K$ sets of 1-D convolutional filters, where the $k$-th set contains $C_k$ filters of width $k$ (i.e. $k$ = 1, 2, . . . , $K$).

입력 sequence는 먼저 $K$개의 1D Convolution 필터들의 집합과 연산된다. 여기서 $k$ 번째 집합은 $C_k$ 개의 너비가 $k$인 필터를 가지고 있다.

These filters explicitly model local and contextual information (akin to modeling unigrams, bigrams, up to K-grams).

이 필터들은 명시적으로 지역적, 문맥적 정보를 추출하는 것을 모델링한다. (unigram, bigram, k-gram까지 모델링하는 것과 유사)

The convolution outputs are stacked together and further max pooled along time to increase local invariances.

Convolution의 출력은 함께 쌓이고, local invariance를 증가시키기 위해 max pooling을 진행한다.

Note that we use a stride of 1 to preserve the original time resolution.

여기서 우리는 stride=1 으로 설정하여 원본의 시간 resolution을 유지한 것에 주목해야 한다.

We further pass the processed sequence to a few fixed-width 1-D convolutions, whose outputs are added with the original input sequence via residual connections (He et al., 2016).

우리는 처리된 sequence를 몇 개의 고정된 너비를 갖는 1D Convolution에 전달하고, 출력은 residual connection을 통해 입력 sequence와 더해지게 된다.

Batch normalization (Ioffe & Szegedy, 2015) is used for all convolutional layers.

배치 정규화는 모든 Convolution 레이어에 적용되었다.

The convolution outputs are fed into a multi-layer highway network to extract high-level features.

Convolution 출력은 다시 다층 highway 신경망의 입력으로 들어가고, 고차원적 특징을 추출한다.

Finally, we stack a bidirectional GRU RNN on top to extract sequential features from both forward and backward context.

마지막으로 우리는 양방향 GRU RNN을 최상단에 배치하여 앞뒤 문맥에서 연속적 속성(sequential feature)을 추출하게 만들었다.

CBHG is inspired from work in machine translation (Lee et al., 2016), where the main differences from Lee et al. (2016) include using non-causal convolutions, batch normalization, residual connections, and stride=1 max pooling.

CBHG는 기계번역에서 영감을 받았고, 둘의 차이는 non-causal convolution을 사용한 것, 배치 정규화를 사용한 것, residual connection과 stride=1 max pooling을 적용한 것이다.

*non-causal convolution: convolution 연산에서 timestep $t+k$ 의 시점의 데이터도 사용 (미래시점도 사용)

We found that these modifications improved generalization.

우리는 위의 수정사항들이 일반화 성능에 기여하였다는 것을 발견하였다.

## 3.2 Encoder

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d23f5dd9-8f86-4fe5-b474-368de3883f05/Untitled.png)

The goal of the encoder is to extract robust sequential representations of text.

Encoder의 목표는 텍스트의 robust한 sequential representation을 추출하는 것이다. (텍스트의 특징 추출)

The input to the encoder is a character sequence, where each character is represented as a one-hot vector and embedded into a continuous vector.

encoder의 입력은 문자열이며, 각 문자들은 원핫벡터로 구성된 이후 continuous 벡터로 임베딩된 것들이다.

We then apply a set of non-linear transformations, collectively called a “pre-net”, to each embedding.

우리는 그러고 나서 각 임베딩에 비선형 변환들을 적용시켰으며 이 비선형 변환 집합을 “pre-net”이라고 불렀다.

We use a bottleneck layer with dropout as the pre-net in this work, which helps convergence and improves generalization.

우리는 본 논문에 등장하는 pre-net에 dropout과 함께 bottleneck 레이어를 사용하였으며 이는 수렴과 일반화 성능을 높이는데 도움을 주었다.

A CBHG module transforms the prenet outputs into the final encoder representation used by the attention module.

CBHG 모듈은 prenet의 출력을 attention 모듈이 사용하는 최종 encoder representation으로 변환시켜준다.

We found that this CBHG-based encoder not only reduces overfitting, but also makes fewer mispronunciations than a standard multi-layer RNN encoder (see our linked page of audio samples).

우리는 이 CBHG기반 인코더가 오버피팅을 줄여줄 뿐만 아니라, 표준 다층 RNN인코더에 비해 발음을 잘못하는 경우를 더 줄여줌을 확인하였다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2b8f77c6-3e54-448f-a969-2f30aa7259ea/Untitled.png)

## 3.3 Decoder

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8e15365e-ebc8-4e52-bef0-7ab15f3886d9/Untitled.png)

We use a content-based tanh attention decoder (see e.g. Vinyals et al. (2015)), where a stateful recurrent layer produces the attention query at each decoder time step.

우리는 컨텐츠 기반 tanh attention decoder를 사용하였으며, 여기서 Stateful Recurrent 레이어는 각 decoder time step에서 attention 쿼리를 생성한다.

We concatenate the context vector and the attention RNN cell output to form the input to the decoder RNNs.

우리는 context 벡터와 attention RNN의 출력을 이어붙여 decoder RNN의 입력으로 사용되는 형태로 만들어주었다.

We use a stack of GRUs with vertical residual connections (Wu et al., 2016) for the decoder.

우리는 decoder에서 수직 residual connection과 함께 GRU를 쌓은 형태를 사용하였다.

We found the residual connections speed up convergence.

우리는 residual connection이 수렴속도를 빠르게 하는 것에 도움을 준다는 것을 발견하였다.

The decoder target is an important design choice.

decoder의 대상은 설계하는데 있어서 중요한 선택이다.

While we could directly predict raw spectrogram, it’s a highly redundant representation for the purpose of learning alignment between speech signal and text (which is really the motivation of using seq2seq for this task).

우리가 raw spectrogram을 바로 예측할 수 있지만, 이는 음성 신호와 텍스트간의 alignment를 학습하기 위한 목적으로는 매우 불필요한 representation이다.

Because of this redundancy, we use a different target for seq2seq decoding and waveform synthesis.

이런 불필요함 때문에 우리는 seq2seq decoding과 waveform 합성에 대해 다른 목적을 설정하였다.

The seq2seq target can be highly compressed as long as it provides sufficient intelligibility and prosody information for an inversion process, which could be fixed or trained.

seq2seq 타겟은 고정되거나 훈련될 수 있는 inversion process에 대해 충분한 정보와 운율 정보가 제공되는 한, 고도로 압축이 가능하다.

We use 80-band mel-scale spectrogram as the target, though fewer bands or more concise targets such as cepstrum could be used.

우리는 80묶음의 mel-spectrogram을 타겟으로 사용하였지만, 더 적은 묶음이나 더 간결한 cepstrum같은 것을 타겟으로 사용할 수 있다.

We use a post-processing network (discussed below) to convert from the seq2seq target to waveform.

우리는 아래에서 이야기할 후처리 네트워크를 통해 seq2seq 타겟(mel-spectrogram)을 waveform으로 변환하였다.

We use a simple fully-connected output layer to predict the decoder targets.

우리는 단순한 FC 출력층을 통해 decoder 타겟을 예측하도록 만들었다.

An important trick we discovered was predicting multiple, non-overlapping output frames at each decoder step.

우리가 발견한 중요한 트릭은 중복되지 않은 여러개의 출력 프레임을 각 decoder step에서 예측하는 것이다.

Predicting $r$ frames at once divides the total number of decoder steps by $r$, which reduces model size, training time, and inference time.

$r$개의 프레임을 한 번에 예측하면, 총 decoder 단계 수가 $r$로 나뉘게 되므로 모델의 크기, 학습 시간, 추론 시간이 줄어들게 된다.

More importantly, we found this trick to substantially increase convergence speed, as measured by a much faster (and more stable) alignment learned from attention.

더 중요한 것은, 우리는 이 트릭이 수렴속도를 크게 향상시킨다는 것이다. 수렴 속도는 attention으로 부터 학습된 alignment를 가지고 측정되었다.

This is likely because neighboring speech frames are correlated and each character usually corresponds to multiple frames.

이것은 인접한 음성 프레임들은 연관되어 있고, 각 문자는 여러개의 프레임에 상응될 수 있기 때문이다.

Emitting one frame at a time forces the model to attend to the same input token for multiple timesteps; emitting multiple frames allows the attention to move forward early in training.

한 번에 하나의 프레임만 내보내는 것은 모델이 여러 timestep 동안 동일한 입력 토큰에 계속 attention을 갖도록 만든다: 여러 프레임을 내보내는 것은 학습 초기에 attention을 앞으로 나아가게 만들어준다.

A similar trick is also used in Zen et al. (2016) but mainly to speed up inference.

유사한 트릭은 2016년에 Zen이 발표한 논문에서도 사용되었지만 주로 추론속도를 향상시키는데 사용되었다.

The first decoder step is conditioned on an all-zero frame, which represents a <GO> frame.

첫 decoder 단계는 모두 0 프레임 상태를 갖고 이는 <GO> 프레임을 나타낸다.

In inference, at decoder step t, the last frame of the r predictions is fed as input to the decoder at step t + 1.

decoder time step $t$의 추론단계에서는 $r$개의 예측에서 마지막 프레임은 다음 time step $t+1$의 입력으로 사용된다.

Note that feeding the last prediction is an ad-hoc choice here – we could use all r predictions.

여기서 예측한 프레임 중 가장 마지막을 사용한 것은 임시의 선택이다. — 여기서 $r$개의 모든 예측 프레임을 사용하는 것도 가능하다.

During training, we always feed every r-th ground truth frame to the decoder.

학습과정에서 우리는 매 $r$번째 ground truth 프레임을 decoder에게 전달한다.

The input frame is passed to a pre-net as is done in the encoder.

입력 프레임은 encoder에서 했던 것 처럼 pre-net에 전달된다.

Since we do not use techniques such as scheduled sampling (Bengio et al., 2015) (we found it to hurt audio quality), the dropout in the pre-net is critical for the model to generalize, as it provides a noise source to resolve the multiple modalities in the output distribution.

우리는 scheduled sampling과 같은 기법을 사용하지 않았기에, pre-net에서 dropout은 모델의 일반화를 위해 중요하며, 이는 출력 분포의 여러 양상을 띄는 것을 해결하기 위해 noise가 포함된 source를 제공한다.

## 3.4 Post-Processing Net and WaveForm Synthesis

As mentioned above, the post-processing net’s task is to convert the seq2seq target to a target that can be synthesized into waveforms.

위에서 언급했던 것처럼, 후처리 신경망의 작업은 seq2seq의 타겟을 waveform으로 합성될 수 있는 target으로 변환하는 것이다.

Since we use Griffin-Lim as the synthesizer, the post-processing net learns to predict spectral magnitude sampled on a linear-frequency scale.

우리는 합성기로 Griffin-Lim을 사용하였기 때문에 후처리 신경망은 선형-주파수 규모에서 샘플링된 spectral 크기를 예측하도록 학습된다.

Another motivation of the post-processing net is that it can see the full decoded sequence.

후처리 신경망의 또다른 동기는 full decoded sequence를 볼 수 있다는 점이다.

In contrast to seq2seq, which always runs from left to right, it has both forward and backward information to correct the prediction error for each individual frame.

항상 왼쪽에서 오른쪽으로 실행되는 seq2seq와 다르게, 후처리 신경망은 각각의 독립된 프레임별로 예측오차를 줄이기 위한 앞과 뒤의 정보가 존재한다.

In this work, we use a CBHG module for the post-processing net, though a simpler architecture likely works as well.

이번 논문에서는 후처리 신경망으로 CBHG 모듈을 사용하였으며, 더 단순한 구조도 잘 작동할 수 있다.

The concept of a post-processing network is highly general.

후처리 신경망의 개념은 매우 일반적(포괄적)이다.

It could be used to predict alternative targets such as vocoder parameters, or as a WaveNet-like neural vocoder (van den Oord et al., 2016; Mehri et al., 2016; Arik et al., 2017) that synthesizes waveform samples directly.

후처리 신경망은 vocoder 파라미터를 예측하는데 사용하거나 Waveform의 샘플을 직접 합성하는 WaveNet과 같은 neural vocoder로도 사용할 수 있다.

We use the Griffin-Lim algorithm (Griffin & Lim, 1984) to synthesize waveform from the predicted spectrogram.

우리는 Griffin-Lim 알고리즘을 사용하여 예측한 spectrogram으로 부터 waveform을 합성하였다.

We found that raising the predicted magnitudes by a power of 1.2 before feeding to Griffin-Lim reduces artifacts, likely due to its harmonic enhancement effect.

우리는 Griffin-Lim에 입력으로 넣기전, 예측한 크기를 1.2배 증가시키게 되면 화음 강화 효과 때문에 artifact가 감소한다는 것을 발견하였다.

We observed that Griffin-Lim converges after 50 iterations (in fact, about 30 iterations seems to be enough), which is reasonably fast.

우리는 Griffin-Lim이 50번의 반복 이후에 수렴한다는 것을 발견했으며 (사실, 30번의 반복도 충분해 보인다.) 이는 매우 빠르다는 것을 보여준다.

We implemented Griffin-Lim in TensorFlow (Abadi et al., 2016) hence it’s also part of the model.

우리는 Griffin Lim을 TensorFlow에 구현하였으므로, 모델의 일부이다.

While Griffin-Lim is differentiable (it does not have trainable weights), we do not impose any loss on it in this work.

Griffin-Lim은 미분이 가능하기 대문에 우리는 어떠한 loss 계산을 부여하지는 않았다.

We emphasize that our choice of Griffin-Lim is for simplicity;

우리가 Griffin-Lim을 사용한 것은 단순함을 강조하기 위해서이다.

while it already yields strong results, developing a fast and high-quality trainable spectrogram to waveform inverter is ongoing work.

이미 강력한 결과를 보여주고 있지만, 빠르고 고품질의 학습 가능한 spectrogram을 waveform으로 변환하는 inverter는 작업이 진행중이다.

# 4. Model Details

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b5928022-d290-488f-8400-26019db0d010/Untitled.png)

Table 1 lists the hyper-parameters and network architectures.

Table 1은 사용한 하이퍼파라미터와 신경망 구조를 나열한 것이다.

We use log magnitude spectrogram with Hann windowing, 50 ms frame length, 12.5 ms frame shift, and 2048-point Fourier transform.

우리는 Hann windowing과 함께 로그가중 spectrogram을 사용하였고, 50ms 프레임 길이, 12.5ms 프레임 이동, 2048-point의 푸리에 변환을 적용시켰다.

We also found pre-emphasis (0.97) to be helpful.

또한 우리는 pre-emphasis로 0.97이 효과적이라는 것을 발견하였다

We use 24 kHz sampling rate for all experiments. 우리는 24kHz sampling rate를 모든 실험에 적용하였다.

We use r = 2 (output layer reduction factor) for the MOS results in this paper, though larger r values (e.g. r = 5) also work well.

우리는 최종 MOS결과를 위해서는 $r=2$ 로 설정하였지만, 더 큰 $r$ 값에 대해서도 잘 동작하였다.

We use the Adam optimizer (Kingma & Ba, 2015) with learning rate decay, which starts from 0.001 and is reduced to 0.0005, 0.0003, and 0.0001 after 500K, 1M and 2M global steps, respectively.

우리는 learning rate decay를 적용한 Adam optimizer를 사용하였으며, learning reate의 시작은 0.001로 시작하여 0.0005, 0.0003, 0.0001로 각각 500K번, 1M번, 2M번의 iteration이후 감소하였다.

We use a simple $\ell$1 loss for both seq2seq decoder (mel-scale spectrogram) and post-processing net (linear-scale spectrogram).

우리는 seq2seq decoder와 후처리 신경망에대해 둘다 간단한 $\ell$1-Loss 를 적용하였다.

The two losses have equal weights.

두 Loss모두 동일한 가중치를 주었다.

We train using a batch size of 32, where all sequences are padded to a max length.

우리는 batch size로 32를 설정하였으며, 각 sequence는 sequence들 중 최대 길이에 맞게 padding을 부여하였다.

It’s a common practice to train sequence models with a loss mask, which masks loss on zero-padded frames.

zero-padding 프레임에서 loss를 마스킹하는 loss mask를 사용하여 sequence model을 학습하는 것이 일반적이다.

However, we found that models trained this way don’t know when to stop emitting outputs, causing repeated sounds towards the end.

하지만, 이렇게 하면 모델이 언제 출력을 멈춰야 하는지 몰라 마지막에 반복된 소리를 낸다는 것을 발견하였다.

One simple trick to get around this problem is to also reconstruct the zero-padded frames.

이 문제를 해결하는 간단한 트릭은 zero-padded 프레임을 재구성하는 것이다.

# 5. Experiments

We train Tacotron on an internal North American English dataset, which contains about 24.6 hours of speech data spoken by a professional female speaker.

우리는 Tacotron을 internal North American English데이터셋을 사용하여 학습시켰고, 데이터셋은 전문 여성 발화자가 녹음한 24.6시간의 음성 데이터로 구성되어 있다.

The phrases are text normalized, e.g. “16” is converted to “sixteen”.

각 단어들은 전처리를 적용하였다. 16을 “십육”으로 변경

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/61768975-5831-479e-b868-e833a18d944d/Untitled.png)

## 5.1 Ablation Analysis

We conduct a few ablation studies to understand the key components in our model.

우리는 모델의 주요 요소들을 파악하기 위해 절제 학습을 수행하였다.

As is common for generative models, it’s hard to compare models based on objective metrics, which often do not correlate well with perception (Theis et al., 2015).

일반적으로 생성 모델의 경우처럼 객관적인 측정 기준을 기반으로 모델을 비교하기가 어렵다. 또한 이는 종종 인식(perception)과는 상관되지 않는다.

We mainly rely on visual comparisons instead.

그대신 우리는 주로 시각적 비교에 의존하였다.

We strongly encourage readers to listen to the provided samples.

우리는 독자들에게 제공된 sample을 들어볼 것을 강력히 권고한다.

First, we compare with a vanilla seq2seq model.

먼저 일반 seq2seq 모델과 비교하였다.

Both the encoder and decoder use 2 layers of residual RNNs, where each layer has 256 GRU cells (we tried LSTM and got similar results).

encoder와 decoder 둘다 2개의 residual RNN 레이어 2개를 사용하였고, 각 레이어는 256개의 GRU cell을 갖고 있다. (LSTM에 대해서도 실험해보았으며 비슷한 결과를 보여주었다.)

No pre-net or post-processing net is used, and the decoder directly predicts linear-scale log magnitude spectrogram.

pre-net과 후처리 신경망을 사용하지 않았고, decoder는 linear-scale log magnitude spectrogram을 직접 예측하도록 하였다.

We found that scheduled sampling (sampling rate 0.5) is required for this model to learn alignments and generalize.

우리는 이 모델에서 alignment와 일반화를 위해 scheduled sampling을 사용해야 함을 발견했다.

We show the learned attention alignment in Figure 3.

우리는 학습된 attention alignment를 Figure 3에 보였다.

Figure 3(a) reveals that the vanilla seq2seq learns a poor alignment.

Figure 3의 (a)는 일반 seq2seq 모델은 alignment를 잘 학습하지 못함을 보여준다.

One problem is that attention tends to get stuck for many frames before moving forward, which causes bad speech intelligibility in the synthesized signal.

한 가지 문제점은 attention이 여러 프레임에서 다음 프레임으로 넘어가기 전 막힌다는 것이고 이는 합성된 음성에서 좋지 않은 소리를 들려주게 된다.

The naturalness and overall duration are destroyed as a result.

따라서 자연스러움과 전반적인 길이가 파괴되어서 결과가 나올 것이다.

In contrast, our model learns a clean and smooth alignment, as shown in Figure 3(c).

반면 Figure 3의 (c)는 우리의 Tacotron 모델은 깨끗하고, 부드러운 alignment를 보여준다.

Second, we compare with a model with the CBHG encoder replaced by a 2-layer residual GRU encoder.

두 번째로, 우리는 Tacotron모델을 CBHG 인코더를 2레이어로 구성된 residual GRU encoder로 대체하였을 때와 비교해보았다.

The rest of the model, including the encoder pre-net, remain exactly the same.

그 이외의 모델의 다른 부분들은 유지하였다. (encoder의 pre-net 역시 동일하게 유지)

Comparing Figure 3(b) and 3(c), we can see that the alignment from the GRU encoder is noisier.

Figure 3의 (b)와 (c)를 비교하였을 때, 우리는 GRU encoder를 사용한 alignment에 조금 더 noise가 있는 것을 확인하였다.

Listening to synthesized signals, we found that noisy alignment often leads to mispronunciations.

실제 합성된 신호를 들어보면, 우리는 noisy한 alignment가 잘 못 발음하도록 이끈다는 것을 보았다.

The CBHG encoder reduces overfitting and generalizes well to long and complex phrases.

이는 CBHG encoder가 오버피팅을 줄이고 길고 복잡한 구문에 대해 일반화 성능이 좋다는 것을 보여준다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9e90605b-5e09-47f0-9eda-dc1f64b9035a/Untitled.png)

Figures 4(a) and 4(b) demonstrate the benefit of using the post-processing net.

Figure 4의 (a), (b)는 후처리 신경망을 사용하는 것의 장점을 보여준다.

We trained a model without the post-processing net while keeping all the other components untouched (except that the decoder RNN predicts linear-scale spectrogram).

우리는 다른 구성요소들은 가만히 둔 채 후처리 신경망만 사용하지 않은 모델을 학습시켜보았다.

With more contextual information, the prediction from the post-processing net contains better resolved harmonics (e.g. higher harmonics between bins 100 and 400) and high frequency formant structure, which reduces synthesis artifacts.

상황에 맞는 정보들이 더 많이 제공되는 경우, 후처리 신경망의 예측은 더 좋은 해상력을 갖는 harmonic과 합성 artifact를 감소시키는 고주파 formant 형태를 보여준다.

## 5.2 Mean Opinion Score Tests

We conduct mean opinion score tests, where the subjects were asked to rate the naturalness of the stimuli in a 5-point Likert scale score.

우리는 MOS 테스트를 진행하였고, 이는 평가자들은 주어진 음성에 대해 최대 5점을 부여하는 테스트이다.

The MOS tests were crowdsourced from native speakers.

MOS 테스트는 원어민들을 대상으로 모집하였다.

100 unseen phrases were used for the tests and each phrase received 8 ratings.

100개의 학습하지 않은 구문들이 테스트에 사용되었고, 각 구문마다 8개의 평가를 받았다.

When computing MOS, we only include ratings where headphones were used.

MOS를 계산할 때, 오직 헤드폰을 사용해 평가한 것만 포함시켰다.

We compare our model with a parametric (based on LSTM (Zen et al., 2016)) and a concatenative system (Gonzalvo et al., 2016), both of which are in production.

우리의 모델을 생성과 관련된 parametric system과 concatenative system을 사용해 비교하였다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2d9a760c-e9d7-4b1d-816b-ee4b693eff30/Untitled.png)

As shown in Table 2, Tacotron achieves an MOS of 3.82, which outperforms the parametric system.

Table 2를 보면 알 수 있듯이, Tacotron은 MOS 3.82점을 달성하였고 이는 parametric system을 능가하는 수준이다.

Given the strong baselines and the artifacts introduced by the Griffin-Lim synthesis, this represents a very promising result.

강력한 baseline과 Griffin-Lim 합성을 사용한 결과물을 비교하였을 때, 이것은 매우 유망한 결과를 보여준다.

# 6. Discussions

We have proposed Tacotron, an integrated end-to-end generative TTS model that takes a character sequence as input and outputs the corresponding spectrogram.

우리는 문자를 입력으로 받아 출력으로 그에 상응하는 spectrogram을 생성하는 end-to-end TTS 모델인 Tacotron을 제안하였다.

With a very simple waveform synthesis module, it achieves a 3.82 MOS score on US English, outperforming a production parametric system in terms of naturalness.

매우 간단한 waveform 합성 모듈을 사용하였음에도 3.82 MOS 점수를 달성하였고, 자연스러움 측면에서 parametric system을 능가하는 점수를 받았다.

Tacotron is frame-based, so the inference is substantially faster than sample-level autoregressive methods.

Tacotron은 프레임을 기반으로 하였기에 추론은 sample 단위의 autoregressive 방법보다 훨씬 빠르게 진행된다.

Unlike previous work, Tacotron does not need handengineered linguistic features or complex components such as an HMM aligner.

이전과 다르게 tacotron은 언어학적 특징을 추출하기 위해 사람의 손을 탈필요가 없고, HMM aligner와 같은 복잡한 구성요소를 사용할 필요도 없다.

It can be trained from scratch with random initialization.

Tacotron은 임의로 초기화된 상태인 밑바닥부터 학습될 수 있다.

We perform simple text normalization, though recent advancements in learned text normalization (Sproat & Jaitly, 2016) may render this unnecessary in the future.

우리는 간단한 텍스트 정규화를 진행하였지만, 최근 발전에 따르면 미래에는 이것을 불필요하게 만들 수도 있다.

We have yet to investigate many aspects of our model; many early design decisions have gone unchanged.

우리는 아직 모델의 많은 측면을 조사하지 못했다; 많은 초기의 설계 결정들이 변경되지 않았다.

Our output layer, attention module, loss function, and Griffin-Lim-based waveform synthesizer are all ripe for improvement.

우리의 출력 레이어, attention 모듈, 손실함수 그리고 Griffin-Lim 기반 waveform 합성 모두 개선의 시기가 무르익었다.

For example, it’s well known that Griffin-Lim outputs may have audible artifacts.

예를 들어, Griffin-Lim의 출력이 들을 만한 결과를 낸다는 것은 잘 알려져있다.

We are currently working on fast and high-quality neural-network-based spectrogram inversion.

우리는 현재 더 빠르고 고품질의 신경망 기반 spectrogram inversion 방법에 대해 연구하고 있다.