import torch
import torch.nn as nn
from transformers import RobertaModel


class ImprovedModel(nn.Module):
    def __init__(self, num_token_labels, num_sentence_labels = 2, lstm_hidden_dim=128, lstm_layers=1, dropout_prob=0.1):
        """
        Args:
            num_token_labels: 트리거 스팬 매핑(토큰/스팬 수준) 분류 클래스 수
            num_sentence_labels: 문장 수준 태스크(트리거 존재 여부/개수 예측) 분류 클래스 수
            lstm_hidden_dim: BiLSTM의 hidden dimension 크기
            lstm_layers: BiLSTM 레이어 수
            dropout_prob: 드롭아웃 확률
        """
        super(ImprovedModel, self).__init__()
        # 문장 인코더: RoBERTa로 전체 문맥을 반영한 토큰 임베딩 및 문장 임베딩 생성
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # 토큰 인코더: RoBERTa의 토큰 임베딩에 순차적 정보를 보완하기 위한 양방향 BiLSTM
        self.token_encoder = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Hierarchical Attention:
        # 문장 인코더의 출력(문장 임베딩)을 토큰 인코더의 출력(토큰 임베딩)과 결합하여,
        # 단순 토큰 순차 정보 외에도 문장 전체의 문맥 정보를 활용해 각 토큰 또는 n-gram 후보에 가중치를 부여
        # 결합: RoBERTa 토큰 임베딩 + BiLSTM 출력 + 확장된 문장 임베딩
        combined_dim = self.roberta.config.hidden_size + 2 * lstm_hidden_dim + self.roberta.config.hidden_size
        self.attention_linear = nn.Linear(combined_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)

        # 토큰 수준 분류기: 최종 트리거 스팬 매핑
        self.token_classifier = nn.Linear(self.roberta.config.hidden_size, num_token_labels)
        # 문장 수준 분류기: 트리거 존재/개수 예측 (문장 전체의 임베딩 사용)
        self.sentence_classifier = nn.Linear(self.roberta.config.hidden_size, num_sentence_labels)

    def forward(self, input_ids, attention_mask):
        # [문장 인코더] RoBERTa를 통해 문장 내 토큰 임베딩을 추출
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        # 문장 임베딩: 첫 토큰(또는 다른 pooling 기법)을 사용하여 문장 전체의 컨텍스트 표현 생성
        sentence_embedding = sequence_output[:, 0, :]  # (batch, hidden_size)

        # [토큰 인코더] BiLSTM을 적용하여 순차적 토큰 정보를 보완
        lstm_out, _ = self.token_encoder(sequence_output)  # (batch, seq_len, 2*lstm_hidden_dim)

        # 문장 임베딩을 각 토큰 위치에 맞게 확장
        sentence_expanded = sentence_embedding.unsqueeze(1).expand(-1, sequence_output.size(1),
                                                                   -1)  # (batch, seq_len, hidden_size)

        # 세 가지 정보를 결합: RoBERTa 토큰 임베딩, BiLSTM 출력, 확장된 문장 임베딩
        combined_features = torch.cat([sequence_output, lstm_out, sentence_expanded],
                                      dim=-1)  # (batch, seq_len, combined_dim)

        # [Hierarchical Attention] 결합된 특성을 통해 각 토큰 또는 n-gram 후보에 대한 가중치 산출
        attn_weights = self.attention_linear(combined_features)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)

        # 원래의 토큰 임베딩에 attention 가중치를 적용해 보정
        enhanced_tokens = sequence_output * attn_weights  # (batch, seq_len, hidden_size)
        enhanced_tokens = self.dropout(enhanced_tokens)

        # [출력 계층]
        # 토큰 수준 예측: 트리거 스팬 매핑
        token_logits = self.token_classifier(enhanced_tokens)  # (batch, seq_len, num_token_labels)
        # 문장 수준 예측: 트리거 존재 여부 및 개수 예측
        sentence_logits = self.sentence_classifier(sentence_embedding)  # (batch, num_sentence_labels)

        return {"token_logits": token_logits, "sentence_logits": sentence_logits}