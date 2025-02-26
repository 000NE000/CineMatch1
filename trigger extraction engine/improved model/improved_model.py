import torch
import torch.nn as nn
from transformers import RobertaModel

class HierarchicalAttentionModel(nn.Module):

    def __init__(self, num_labels):
        super(HierarchicalAttentionModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

        # 문장 임베딩과 토큰 임베딩을 결합하기 위한 projection layer 및 attention vector
        self.sentence_projector = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)
        self.attention_vector = nn.Parameter(torch.randn(self.roberta.config.hidden_size))

    def compute_sentence_embedding(self, token_embeddings, sentence_masks):
        """
        token_embeddings: [batch, seq_length, hidden_size]
        sentence_masks: [batch, num_sentences, seq_length] → 각 문장이 해당하는 토큰 위치 (0/1 mask)
        """
        # 마스킹된 토큰 임베딩에 대해 각 문장별 평균 풀링
        token_embeddings_expanded = token_embeddings.unsqueeze(1).expand(-1, sentence_masks.size(1), -1, -1)
        sentence_masks_expanded = sentence_masks.unsqueeze(-1)  # [B, num_sentences, seq_length, 1]

        masked_embeddings = token_embeddings_expanded * sentence_masks_expanded
        sentence_sum = masked_embeddings.sum(dim=2)  # [B, num_sentences, hidden_size]
        mask_sum = sentence_masks_expanded.sum(dim=2)
        sentence_embedding = sentence_sum / (mask_sum + 1e-8)
        return sentence_embedding  # [B, num_sentences, hidden_size]

    def hierarchical_attention(self, token_embeddings, sentence_embeddings, sentence_boundaries):
        """
        sentence_boundaries: [batch, seq_length] 정수값 (각 토큰이 속하는 문장 인덱스)
        """
        batch_size, seq_length, hidden_size = token_embeddings.size()
        # 각 토큰에 대해 해당 문장의 임베딩을 가져옵니다.
        token_sentence_embeds = []
        for i in range(batch_size):
            # sentence_boundaries[i]: [seq_length] 각 토큰이 속하는 문장 인덱스 (예: 0 ~ num_sentences-1)
            token_sentence_embed = sentence_embeddings[i][sentence_boundaries[i]]
            token_sentence_embeds.append(token_sentence_embed)
        token_sentence_embeds = torch.stack(token_sentence_embeds, dim=0)  # [B, seq_length, hidden_size]

        # 토큰 임베딩과 문장 임베딩을 단순 합산 후 projector 적용
        combined = token_embeddings + token_sentence_embeds
        combined = self.sentence_projector(combined)

        # attention vector와 dot product를 통해 각 토큰에 가중치 부여
        attn_weights = torch.matmul(combined, self.attention_vector)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)  # [B, seq_length, 1]

        # 업데이트된 토큰 임베딩 (Residual 연결)
        attended = token_embeddings * attn_weights
        updated_token_embeddings = token_embeddings + attended
        return updated_token_embeddings

    def forward(self, input_ids, attention_mask, sentence_masks, sentence_boundaries, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, seq_length, hidden_size]

        # 문장 임베딩 계산
        sentence_embeddings = self.compute_sentence_embedding(token_embeddings, sentence_masks)
        # hierarchical attention 적용
        updated_token_embeddings = self.hierarchical_attention(token_embeddings, sentence_embeddings,
                                                               sentence_boundaries)
        updated_token_embeddings = self.dropout(updated_token_embeddings)

        logits = self.token_classifier(updated_token_embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return loss, logits