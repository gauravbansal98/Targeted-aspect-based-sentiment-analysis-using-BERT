# ABSA as a Sentence Pair Classification Task

## Requirement

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4
* nltk
* sklearn

## Step 1: prepare datasets

### SentiHood

Since the link given in the [dataset released paper](<http://www.aclweb.org/anthology/C16-1146>) has failed, we use the [dataset mirror](<https://github.com/uclmr/jack/tree/master/data/sentihood>) listed in [NLP-progress](https://github.com/sebastianruder/NLP-progress/blob/master/english/sentiment_analysis.md) and fix some mistakes (there are duplicate aspect data in several sentences). See directory: `data/sentihood/`.

I changed the sentence pair task to a single sentence task where we add the aspect after the location and then we train the model to predict from {None, positive, negative}

Run following commands to prepare datasets for tasks:

```
cd generate/
python3 generate_sentihood.py
```


## Step 2: train

We have provided different models for using bert use bert_model.py, for using distilbert use model.py

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
python model.py --num_epochs 10 --max_seq_length 512


## Step 3: evaluation

Evaluate the results on test set (calculate Acc, F1, etc.).

```
python evaluation.py  --pred_data_dir results/test_ep_4.txt

