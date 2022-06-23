import pandas as pd
from Config import Config
from load_data import traindataloader, idx2label, devdataloader
from models1 import Bert_Ner
import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import time

SEED = Config.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else 'cpu'
SAVED_DIR = '/xiaowang/ner_dataset/shiyan1/Flat-Wubi/saved_model/model5'

N_EPOCHS = Config.epoch
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1

model = Bert_Ner(idx2label=idx2label, use_bert=Config.use_bert, use_wubi=Config.use_wubi,
                 use_bigrams=Config.use_bigrams)
model.to(device)

lr_for_linear = Config.lr_for_linear
lr_for_embedding = Config.lr_for_embedding
lr_for_crf = Config.lr_for_crf
lr_for_attn = Config.lr_for_embedding
lr_for_bert = Config.lr_for_bert
lr_for_wubi = Config.lr_for_wubi
optimizer_grouped_parameters = [
    {'params': model.bert.parameters(), 'lr': lr_for_bert, 'weight_dacay': 0.},
    {'params': model.hidden2label.parameters(), 'lr': lr_for_linear, 'weight_dacay': 0.},
    {'params': model.char_hidden2attn.parameters(), 'lr': lr_for_linear, 'weight_dacay': 0.},
    {'params': model.lex_hidden2attn.parameters(), 'lr': lr_for_linear, 'weight_dacay': 0.},
    {'params': model.crf.parameters(), 'lr': lr_for_crf, 'weight_dacay': 0.},
    {'params': model.attn.parameters(), 'lr': lr_for_attn, 'weight_dacay': 0.},
    {'params': model.attn_for_wubi.parameters(), 'lr': lr_for_attn, 'weight_dacay': 0.},
    {'params': model.attn_for_total.parameters(), 'lr': lr_for_attn, 'weight_dacay': 0.},
    #{'params': model.hidden_two_label.parameters(), 'lr': lr_for_linear, 'weight_dacay': 0.},
    {'params': model.lattice_embedding.parameters(), 'lr': lr_for_embedding, 'weight_dacay': 0.},
    {'params': model.bigrams_embedding.parameters(), 'lr': lr_for_embedding, 'weight_dacay': 0.},
    {'params': model.wubi_embedding.parameters(), 'lr': lr_for_wubi*lr_for_embedding, 'weight_dacay': 0.},
    {'params': model.char_hidden2attn_for_wubi.parameters(), 'lr': lr_for_linear, 'weight_dacay': 0.},
]

total_steps = len(traindataloader) * N_EPOCHS
optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps),
                                            num_training_steps=total_steps)

time1 = time.time()

def extract(chars, tags):
    assert len(chars) == len(tags)
    result = []
    pre = ''
    w = []
    for idx, tag in enumerate(tags):
        if pre == '':
            if tag.startswith('B'):
                pre = tag.split('-')[1]
                w.append(chars[idx])
            if tag.startswith('S'):
                pre = tag.split('-')[1]
                w.append(chars[idx])
                result.append([w, pre])
                w = []
                pre = ''
                continue

            # if idx == len(tags) - 1:
            #     result.append([w, pre])
        else:
            if tag == f'M-{pre}':
                w.append(chars[idx])

            elif tag == f'E-{pre}':
                w.append(chars[idx])
                result.append([w, pre])
                w = []
                pre = ''

            else:
                result.append([w, pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
    return [[''.join(x[0]), x[1]] for x in result]

best_f1 = 0
best_epoch = -1

for epoch in range(N_EPOCHS):
    model.train()
    pbar = tqdm(traindataloader)
    pbar.set_description("[Train Epoch {}]".format(epoch))
    LOSS = 0

    for batch_idx, batch_data in enumerate(pbar):
        model.zero_grad()
        loss = model(batch_data, mode='train')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        LOSS += loss.item()/Config.BATCH_SIZE
    print(LOSS)


    gold_num = 0
    pred_num = 0
    correct_num = 0
    model.eval()
    for idx, batch_data in enumerate(devdataloader):
        pred = model(batch_data, mode='test')[0]
        target = batch_data['target'][0]

        raw_chars = batch_data['raw_chars'][0]
        sentence = raw_chars

        true_labels = [idx2label[i.item()] for i in target]
        true_entities = extract(sentence, true_labels)
        new_true_entities = true_entities[:len(true_entities)]
        gold_num += len(new_true_entities)

        pred_labels = [idx2label[i.item()] for i in pred]
        pred_entities = extract(sentence, pred_labels)
        new_pred_entities = pred_entities[:len(pred_entities)]
        pred_num += len(new_pred_entities)

        for i in new_pred_entities:
            if i in new_true_entities:
                correct_num += 1

    try:
        pre = correct_num / pred_num
        rec = correct_num / gold_num
        f1 = 2 * pre * rec / (pre + rec)
        print('此epoch测试结果为:')
        print('Pre: ', pre)
        print('Rec: ', rec)
        print('F1: ', f1)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            print('得到更优结果，保存模型!')
            model.cpu()
            torch.save(model, SAVED_DIR)
    except:
        print('无结果')
        continue
    model = model.to(device)
    # if epoch > 2:
    #     model.cpu()
    #     torch.save(model, f'/xiaowang/ner_dataset/shiyan1/Flat-Wubi/saved_model/model{epoch}')
    #     model = model.to(device)
    if epoch - best_epoch > 14:
        break



time2 = time.time()

print(f'训练和保存模型一共用了{time2 - time1}秒')
print(f'最优F1为{best_f1}, 在第{best_epoch}个epoch得到')
