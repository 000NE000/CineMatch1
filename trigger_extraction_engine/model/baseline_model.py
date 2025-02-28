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
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)
        return logits