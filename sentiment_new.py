from transformers import DistilBertForSequenceClassification
from torch import nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):

    super(SentimentClassifier, self).__init__()

    self.bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 3)

  def forward(self, input_ids, positions, attention_mask, label_ids):

    emb = self.bert.get_input_embeddings()
    embeds = emb(input_ids)
    pad_emb = emb(torch.tensor([[0]]).to(device)).squeeze(0).squeeze(0)
    new_embeds = []
    for i in range(input_ids.shape[0]):
      pos = [int(j) for j in positions[i].split(' ')[:-1]]
      embed = []
      j = 0
      while (j < embeds[i].shape[0]):
        summ = embeds[i][j]
        while(j in pos[:-1]):
          j += 1
          summ += embeds[i][j]
        embed.append(torch.tensor(summ))
        j += 1
      for i in range(len(pos)-1):
        embed.append(pad_emb)
      new_embeds.append(torch.stack(embed))
    outputs = self.bert(inputs_embeds = torch.stack(new_embeds) , attention_mask=attention_mask, labels=label_ids)
    return outputs