from transformers import DistilBertForSequenceClassification
from torch import nn
import torch

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):

    super(SentimentClassifier, self).__init__()

    self.bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 3)

    self.drop = nn.Dropout(p=0.3)

    self.out = nn.Linear(self.bert.config.hidden_size, 3)

  def forward(self, input_ids, attention_mask, label_ids):

    outputs = self.bert(input_ids, attention_mask=attention_mask, labels=label_ids)
    return outputs
    # pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    # pooled_outputs = []
    # for output in pooled_output:
    #     pooled_outputs.append(torch.FloatTensor(output[0, :]))
    # output = self.drop(torch.FloatTensor(pooled_outputs))

    # return self.out(output)