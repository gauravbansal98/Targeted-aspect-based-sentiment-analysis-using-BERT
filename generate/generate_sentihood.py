import os

from data_utils_sentihood import *

data_dir='../data/sentihood/'
aspect2idx = {
    'general': 0,
    'price': 1,
    'transit-location': 2,
    'safety': 3,
}

(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task(data_dir, aspect2idx)
print("len(train) = ", len(train))
print("len(val) = ", len(val))
print("len(test) = ", len(test))
train.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
val.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
test.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(dir_path+"train_QA_M.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tlabel\tpositions\n")
    for v in train:
        if(v[1].count(v[2]) == 0):
            continue
        f.write(str(v[0])+"\t")
        idx = v[1].index(v[2])
        position = ""
        position = position + str(idx) + " "
        for item in v[3]:
            v[1].insert(idx+1, item)
            position = position + str(idx+1) + " "
            idx += 1
        f.write(' '.join(v[1]) + "\t")
        f.write(v[4] + "\t")
        f.write(position +"\n")
        
       
with open(dir_path+"dev_QA_M.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tlabel\tpositions\n")
    for v in val:
        if(v[1].count(v[2]) == 0):
            continue
        f.write(str(v[0])+"\t")
        idx = v[1].index(v[2])
        position = ""
        position = position + str(idx) + " "
        for item in v[3]:
            v[1].insert(idx+1, item)
            position = position + str(idx+1) + " "
            idx += 1
        f.write(' '.join(v[1]) + "\t")
        f.write(v[4] + "\t")
        f.write(position +"\n")

with open(dir_path+"test_QA_M.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tlabel\tpositions\n")
    for v in test:
        if(v[1].count(v[2]) == 0):
            continue
        f.write(str(v[0])+"\t")
        idx = v[1].index(v[2])
        position = ""
        position = position + str(idx) + " "
        for item in v[3]:
            v[1].insert(idx+1, item)
            position = position + str(idx+1) + " "
            idx += 1
        f.write(' '.join(v[1]) + "\t")
        f.write(v[4] + "\t")
        f.write(position +"\n")