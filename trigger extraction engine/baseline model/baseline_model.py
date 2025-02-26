import torch
import torch.nn as nn
from transformers import RobertaModel


class BaseTokenClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BaseTokenClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # [batch, seq_length, hidden_size]
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten logits and labels for token-wise loss 계산
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return loss, logits