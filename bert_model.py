import pandas as pd
import os
from transformers import BertTokenizer
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from bert_sentiment import SentimentClassifier
from torch import nn
import torch
import numpy as np
from scipy.special import softmax

import logging
import time

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--output_dir", default="results", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--init_checkpoint", default=None, type=str, help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    args = parser.parse_args()
    return args

args = arg_parser()

word_to_idx = {"None" : 0, "Positive" : 1, "Negative" : 2}
data_dir = "data/sentihood/bert-pair/"
data = pd.read_csv(os.path.join(data_dir, "train_QA_M.tsv"), sep="\t")
test_data = pd.read_csv(os.path.join(data_dir, "test_QA_M.tsv"), sep="\t")

test_data["label"] = test_data.label.map(word_to_idx)
data["label"] = data.label.map(word_to_idx)
#ids, sentences, label = (data['id'], data['sentence1'], data['label'])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# input_ids = []
# attention_masks = []
# for sentence in sentences:
#     encoding = tokenizer.encode_plus(sentence, max_length=args.max_seq_length, add_special_tokens=True,
#                                     return_token_type_ids=False,
#                                     pad_to_max_length=True,
#                                     return_attention_mask=True,
#                                     return_tensors='pt',)  # Return PyTorch tensors
#     input_ids.append(encoding['input_ids'])
#     attention_masks.append(encoding['attention_mask'])

class GPReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(review,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_token_type_ids=False,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            )

        return {
        'review_text': review,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(reviews=df.sentence1,
                        targets=df.label,
                        tokenizer=tokenizer,
                        max_len=max_len
                        )

    return DataLoader(ds, batch_size=batch_size)

train_data_loader = create_data_loader(data, tokenizer, args.max_seq_length, args.batch_size)
test_data_loader = create_data_loader(test_data, tokenizer, args.max_seq_length, args.batch_size)

model = SentimentClassifier(3)
if args.init_checkpoint and os.path.exists(args.init_checkpoint):
    model.load_state_dict(torch.load(args.init_checkpoint))
model.to(device)

optimizer = AdamW(model.parameters(), lr=args.learning_rate)

total_steps = len(train_data_loader) * args.num_train_epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

# loss_fn = nn.CrossEntropyLoss().to(device)

best_accuracy = 0
global_step = 0
for epoch_num in range(int(args.num_train_epochs)):
    model.train()
    train_loss, train_accuracy = 0, 0
    nb_train_steps, nb_train_examples = 0, 0

    for step, batch_data in enumerate(train_data_loader):
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        targets = batch_data["targets"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_ids = targets)

        loss, logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        label_ids = targets.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        train_accuracy += np.sum(preds == label_ids)
        train_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        nb_train_examples += input_ids.size(0)
        nb_train_steps += 1
        global_step += 1
        if (step+1)%50==0:
            logger.info("Epoch = %d, Batch = %d, Batch loss = %f, Avg loss (per batch) = %f", 
            epoch_num+1, (step+1), loss.item(), train_loss/(step+1))
    logger.info("Creating a checkpoint.")
    model.eval().cpu()
    ckpt_model_filename = "bert_ckpt_epoch_" + str(epoch_num+1) + ".pth"
    ckpt_model_path = os.path.join(args.output_dir, ckpt_model_filename)
    torch.save(model.state_dict(), ckpt_model_path)
    model.to(device)
    model.eval()
    with open(os.path.join(args.output_dir, "bert_test_ep_"+str(epoch_num+1)+".txt"), "w") as f_test:
        for step, batch_data in enumerate(test_data_loader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            targets = batch_data["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_ids = targets)

            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            logits = softmax(logits, axis=1)
            label_ids = targets.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            for output_i in range(len(outputs)):
                f_test.write(str(outputs[output_i]))
                for ou in logits[output_i]:
                    f_test.write(" "+str(ou))
                f_test.write("\n")

    train_loss /= nb_train_steps
    train_accuracy /= nb_train_examples
    logger.info("After %f epoch, Training loss = %f, Training accuracy = %f", epoch_num+1, train_loss, train_accuracy)


model.eval().cpu()
timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
save_model_filename = "bert_epoch_" + str(args.num_train_epochs) + "_" + timestamp + ".model"
save_model_path = os.path.join(args.output_dir, save_model_filename)
torch.save(model.state_dict(), save_model_path)