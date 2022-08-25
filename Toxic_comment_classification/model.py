import torch
import torch.nn as nn
from transformers import BertModel


class ToxicClassifier(nn.Module):
    """Class with a pretrained BertModel and classifier"""

    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None, bert_model_name=None):
        """Initialize pretrained BertModel"""

        super().__init__()
        self.bert_model_name = bert_model_name
        self.bert = BertModel.from_pretrained(self.bert_model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask):
        """Forward pass"""

        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)

        return output
