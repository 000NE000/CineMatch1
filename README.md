# CineMatch1

# 0x01. 프로젝트 개요 (Overview)
### 1.1 주제와 동기
##### 주요 목표
사용자가 영화로부터 얻을 수 있는 교훈, 인사이트, 혹은 감정적 경험에 초점을 맞춘 추천 시스템을 구축
###### 핵심 접근법
과거 시청 기록이나 장르 중심의 분류에만 의존하지 않고, 각 영화가 제공하는 학습 요소나 느낄 수 있는 감정을 부각해 추천의 의미를 높이고자 함.
###### 사용자 혜택
영화를 선택하기 전에 해당 영화가 어떤 점에서 생각이나 감정에 영향을 줄 수 있는지 미리 확인하게 함으로써, 보다 만족스럽고 목적성 있는 감상 경험을 제공할 수 있음
- 근거 : 콘텐츠 소비가 사용자의 가치관 형성에 영향을 미치며, 삶의 교훈이나 가치관과 연결된 추천 시스템이 더 깊은 몰입 경험을 제공한다 Values and Virtues in Recommender Systems (RecSys Conference, 2021) 
### 1.2 전체 기획
###### 트리거(trigger)의 정의
- 영화 시나리오(또는 자막 스크립트)에서 특정 가치를 느끼게 하는 서사 요소를 ‘트리거’로 명명
	- 중간 수준의 추상화 레벨을 가진 서사 단위를 정의
- 이때 가치는  Schwartz (2012)의 the theory of basic human value에 따른 10가지 요소로 분류한다

| 대분류                | 소분류(단위)                         |
| ------------------ | ------------------------------- |
| Openness to change | self-direction, stimulation     |
| Self-enhancement   | hedonism, achievement, power    |
| Conservation       | security, conformity, tradition |
| Self-transcendence | benevolence, universalism       |

###### recommendation system의 구조
1. **Input System**
	- 영화별 스크립트(혹은 플롯 요약)
	- chatbot을 통한 사용자 선호 표현 파악
2. **Trigger Extraction Engine:**
	- 영화 스크립트에서 감정·가치·교훈을 유발하는 사건(트리거)들을 찾아냄
3. **Value Induction Model:**
	- 해당 영화가 어떤 종류의 가치를 제공하는지(정의, 혁신, 희생 등)를 분류
4. **Recommendation:**
	- 사용자 선호와 트리거-가치 매핑 결과를 기반으로 적합한 영화를 추천

### 1.3 현재 진행 중인 단계 : Trigger Extraction Engine 구축
아래의 내용은 이 단계에 대한 서술이다

---
# 0x02.  **데이터 준비**
### 2.1 시나리오(Scenario) & 플롯(Plot) 데이터
##### scenario data
- VALUENET dataset에 기반함 
	- ValueNet: A New Dataset for Human Value Driven Dialogue System(2022)
- 한 줄로 구성된 짧은 상황(Scenario) 텍스트 + ‘인덕션’ 라벨(해당 상황이 특정 가치를 촉진하는지, 무관한지, 혹은 반대로 훼손하는지)
##### plot data
- 영화 전체의 줄거리(Plot) 요약. 한 편당 여러 문장으로 구성
- wikipedia의 영화 시놉시스, 플롯 설명 data
### 2.2 preprocess pipeline
##### 문장 분리 및 토큰화
- SpaCy 이용하여 문장 단위로 분리
- RoBERTa Tokenizer로 토큰화 → (token, index) 매핑 정보까지 저장
##### Annotation : Zero-Shot Learning
- gpt-4o를 활용한 zero-shot learning 활용
- 다음 프롬프트를 사용
```python
prompt = """  
[input]  
- Extract the triggers from the following scenario, paying attention to the specified Value and its Induction Level:  
- input csv file has columns which are value,scenario,label  
- input data: {csv_text}  
  
[Annotation Guidelines]  
1. The "Value Induction Level" indicates:  
   - 1: the scenario promotes the value.   - 0: the scenario is unrelated to the value.   - -1: the scenario reduces or contradicts the value.2. Definition of a "trigger":  
   - A central event or element in the scenario that drives the narrative and influences the specified value.3. Instructions:  
   - Analyze the entire scenario context to identify any triggers.   - If multiple triggers exist, list each one separately.   - Express each trigger as a abstract, generalized word or phrase(at most 2 words).   [Output Format]  
Return the output strictly in JSON format with the following keys:  
For each row, produce a JSON object with keys "Trigger(s)", "value", and "label". Then return them all in a list. No extra text!  
  
- "Trigger(s)": A list of the identified triggers. If no relevant trigger is found, return ["No relevant event"].  
- "value": The specified value.  
- "label": The specified label.  
"""
```
##### n-gram Embedding & 유사도 기반 매핑
- 트리거 후보(예: “Moral Violation”)가 실제 문장에 정확히 등장하지 않는 경우가 많음
- 따라서, 문장 내 **1-gram ~ 2-gram 후보**를 생성 → RoBERTa 임베딩 계산 → GPT-4가 뽑아준 트리거 임베딩과 **코사인 유사도**로 가장 가까운 표현을 매핑
- 유사도가 기준치(예: 0.8) 이상인 n-gram을 해당 트리거로 간주

##### **길이 제한(Truncation) 이슈 대응**
- 플롯 데이터의 경우, 길이 제한을 고려해야 함
- RoBERTa 입력 길이가 제한되어 있으므로, 너무 긴 문장은 **chunk**로 나누어 처리
- chunk 사이에 걸쳐 있는 트리거를 **merge**하는 로직(Stride)을 통해 **중복·누락 방지**
- 겹치는 청크에서 동일 trigger 병합 파이프라인을 추가해야 함.
##### BIO mapping & 저장
- 위에서 구한 span 정보를 기반으로 BIO tagging
- jsonl 형태로 저장하여  불필요한 재처리를 줄이고, 재현 가능성(reproducibility) 확보
# 0x03. model development
### 3.1 Baseline Model 
##### overview
- RoBERTa **encoder** 위에 **토큰 분류기(선형 레이어)**를 올려, 각 토큰이 B-TRIGGER, I-TRIGGER, O 중 무엇인지를 예측
-  “이 토큰이 트리거의 시작(B), 트리거 내부(I), 트리거가 아님(O)인지” 3가지 태그를 분류
##### 구현
```python
import torch.nn as nn

from transformers import RobertaModel

  

class BaselineModel(nn.Module):

    def __init__(self, num_labels):

        super(BaselineModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

  

    def forward(self, input_ids, attention_mask):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        return logits
```
- loss function : CrossEntropy
- Optimizer: AdamW
- metric : Precision, Recall
	- Recall 값 : 68.1%로 trigger 시작 위치를 놓치는 경우가 상당히 많음
	- Precision : 82.4%로 불필요한 트리거 예측으로 인해 정밀도 떨어짐

### 3.2 개선 방안 탐색
##### 한계점 분석
1. Baseline 모델은 RoBERTa 기반 토큰 분류기로, 모든 토큰을 동일한 수준의 Self-Attention으로 처리하기 때문에 문장 간의 상위-하위 사건 구조나 순차적 흐름(Temporal Dependency)를 충분히 반영하지 못함.
2. **문맥 정보 부족:** 각 토큰의 예측이 전체 문장 또는 문단의 맥락을 충분히 고려하지 못함.
3. **순차 정보 미반영:** Transformer의 self-attention 구조는 순차적 순서를 명시적으로 고려하지 않으므로, 사건의 시작과 끝을 정확하게 잡기 어려움.
##### 조사 및 결론
1. **Hierarchical Attention Network (HAN)** Yang et al. (2016)
	- 여기서는 BiLSTM과 같은 순환 신경망을 결합하는 연구들을 검토
	- BiLSTM을 통한 순차적 정보를 보완하고 이를 Transformer 임베딩과 결합하는 방식은, 사건의 시작과 순차적 관계를 보다 명확히 잡는 데 도움을 줌
2. HAN의 아이디어를 참고하여, 문장 전체의 embedding(첫 토큰 또는 평균 풀링)을 토큰 임베딩과 결합한 후, attention 가중치를 다시 계산하는 방식을 도입하여, 문장 전체 맥락 반영
3. BiLSTM을 활용하여 RoBERTa 토큰 임베딩의 순차적 정보를 보완하고, 이 정보를 문장 임베딩과 함께 결합하는 구조를 설계
4. 이렇게 결합된 정보를 바탕으로 hierarchical attention 메커니즘을 적용하여 최종 토큰 예측을 진행
### 3.3 improved model 
##### idea
- 감정·교훈을 유발하는 트리거는 종종 “특정 단어 하나”가 아니라, **문장의 맥락**과 **이전/이후 사건 전개**에 의존
- 문장 수준 임베딩(시퀀스의 첫 토큰 또는 mean pooling 결과)을 **토큰 임베딩과 결합**해, 각 토큰을 문장 전체 맥락 위에서 재평가
- 최종적으로 “문장 전체 맥락”을 반영한 가중치를 각 토큰 임베딩에 곱해 Token Classifier로 전달
##### model architecture

| 구성요소                       | 역할                                    | 입력 및 처리                                              | 출력 및 설명                                                                                                                       |
| -------------------------- | ------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **입력 레이어**                 | 토큰 인덱스 및 어텐션 마스크 수신                   | `input_ids`, `attention_mask`                        | 원본 텍스트 데이터 (`batch_size`, `seq_len`)                                                                                          |
| **문장 인코더 (RoBERTa)**       | 문맥 기반 토큰 임베딩 및 문장 임베딩 생성              | `RoBERTa(input_ids, attention_mask)`                 | `sequence_output`: (`batch_size`, `seq_len`, `hidden_size`)<br> `sentence_embedding`: (`batch_size`, `hidden_size`) (첫 토큰 활용) |
| **토큰 인코더 (BiLSTM)**        | 토큰 간 순차적 관계 보완                        | `sequence_output`                                    | `lstm_out`: (`batch_size`, `seq_len`, `2*lstm_hidden_dim`)                                                                    |
| **문장 임베딩 확장**              | 문장 임베딩을 각 토큰에 맞게 확장                   | `sentence_embedding.unsqueeze(1).expand()`           | `sentence_expanded`: (`batch_size`, `seq_len`, `hidden_size`)                                                                 |
| **결합 (Concatenation)**     | RoBERTa 임베딩, BiLSTM 출력, 확장된 문장 임베딩 결합 | `sequence_output`, `lstm_out`, `sentence_expanded`   | `combined_features`: (`batch_size`, `seq_len`, `combined_dim`)                                                                |
| **Hierarchical Attention** | 각 토큰 또는 n-gram 후보에 가중치 부여             | `combined_features` → `attention_linear` → `softmax` | `attn_weights`: (`batch_size`, `seq_len`, `1`)<br>`enhanced_tokens`: (`batch_size`, `seq_len`, `hidden_size`)                 |
| **토큰 분류기**                 | 최종 트리거 스팬 매핑 (토큰 수준 예측)               | `enhanced_tokens`                                    | `token_logits`: (`batch_size`, `seq_len`, `num_token_labels`)                                                                 |
| **문장 분류기**                 | 트리거 존재 여부/개수 예측 (문장 수준 예측)            | `sentence_embedding`                                 | `sentence_logits`: (`batch_size`, `num_sentence_labels`)                                                                      |
| **출력 계층**                  | 다중 태스크 결과 반환                          | -                                                    | `{"token_logits": token_logits, "sentence_logits": sentence_logits}`                                                          |
# 0x04 실험 및 결과
### 4.1 성능 비교
- **데이터셋**: 전처리 후 BIO 태깅된 시나리오 + 일부 플롯 문장

|          | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Baseline | 82%       | 68%    | 74%      |
| improved | 88%       | 78%    | 83%      |
- 분석 : Hierarchical Attention을 통해 문맥 전체를 반영하게 되면서, **트리거 시작 위치**를 놓치는 사례가 줄고, **불필요한 위치**를 트리거로 잘못 분류하는 비율도 개선됨

### 4.2 결론 및 개선 
- 트랜스포머 기반 단순 Token Classification(Baseline)에서도 어느 정도 가능성을 확인했지만, 문장 단위 계층 정보를 반영하는 Hierarchical Attention 기법 도입 시 **정확도 및 F1**이 크게 개선됨
