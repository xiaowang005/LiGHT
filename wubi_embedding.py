import sys
from Config import Config

WUBI = {}
wubi2idx = {}
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import label2idx
from fastNLP import seq_len_to_mask
device = "cuda" if torch.cuda.is_available() else 'cpu'

with open(r'/xiaowang/ner_dataset/shiyan1/wubi.txt', encoding='utf-8', mode='r') as f:
    ff = f.readlines()
    for pair in ff:
        pair = pair.replace('\n', '').split(' ')
        if '|' in pair[1]:
            pair[1] = pair[1].split('|')[1]
            WUBI[pair[0]] = pair[1]
        else:
            WUBI[pair[0]] = pair[1]
        for word in pair[1]:
            if word not in wubi2idx:
                wubi2idx[word] = len(wubi2idx)
wubi2idx['z'] = len(wubi2idx)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[-1])

# class TextCNN(nn.Module):
#     def __init__(self, embed_size, kernel_sizes, num_channels, label2idx=label2idx, hidden_size=50):
#         super(TextCNN, self).__init__()
#         self.convs = nn.ModuleList()
#         self.label_size = len(label2idx)
#         self.hidden_size = hidden_size
#         #self.init_embedding = nn.Parameter(self.init_embedding, requires_grad=False)
#         for k, n in zip(kernel_sizes, num_channels):
#             self.convs.append(nn.Conv1d(in_channels=embed_size,
#                                         out_channels=n,
#                                         kernel_size=k, bias=True, padding=k//2
#                                         ))
#         self.linear = nn.Linear(sum(num_channels), self.hidden_size)
#         self.pool = GlobalMaxPool1d()
#
#         self.embedding = nn.Embedding(26, embed_size)
#
#     def forward(self, input, real_len):
#         input = input.long() # batch, seq_len, bihua
#
#         input = self.embedding(input) # batch, seq_len, bihua, embed_size
#         #input = self.dropout(input)
#         #input = get_wordvec(input, self.init_embedding)
#
#         batch, max_seq_len, max_bihua, embed = input.size()
#         input_mask = seq_len_to_mask(torch.tensor(real_len))
#         input_mask = input_mask.to(device)
#         input = input.masked_fill(~(input_mask).unsqueeze(-1).unsqueeze(-1), 0)
#
#         input = input.reshape(batch*max_seq_len, max_bihua, -1)
#         input = input.transpose(1, 2) # res batch*seq_len, out_channels, max_bihua
#
#         input = torch.cat([self.pool(F.relu(conv(input))).squeeze(-1) for conv in self.convs], dim=-1)
#         input = input.reshape(batch, max_seq_len, -1)
#         input = self.linear(input)
#         #input = self.dropout(input)
#
#         return input

class TextCNN(nn.Module):
    def __init__(self, embed_size, kernel_sizes, num_channels, label2idx=label2idx, hidden_size=50):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList()
        self.label_size = len(label2idx)
        self.hidden_size = hidden_size

        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for k, n in zip(kernel_sizes, num_channels):
            self.convs.append(nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(2, k, embed_size-1),
                                        padding=(0, k//2, (embed_size-1)//2) ))
            self.convs.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(k, embed_size-1),
                                        padding=(k//2, (embed_size-1)//2) ))
            self.convs.append(nn.Conv1d(in_channels=16, out_channels=n, kernel_size=k, padding=k//2))

        self.embedding_wubi = nn.Embedding(26, embed_size)
        self.embedding_zhengma = nn.Embedding(27, embed_size)

        self.hidden2embed = nn.Linear(sum(num_channels), embed_size)
        self.dropout = nn.Dropout(0.2)

        # self.test1 = nn.Conv3d(in_channels=1, out_channels=100, kernel_size=(2, 5, embed_size-1),
        #                        padding=(0, 5//2, 24))
        # self.test2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(5, embed_size-1),
        #                        padding=(5//2, 24))
        # self.test3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=5, padding=5//2)

    def forward(self, input_wubi, input_zhengma, real_lens):
        input_wubi = input_wubi.long()
        input_zhengma = input_zhengma.long()

        input_wubi = self.embedding_wubi(input_wubi)
        input_zhengma = self.embedding_zhengma(input_zhengma)
        mask = seq_len_to_mask(torch.tensor(real_lens))
        mask = mask.to(device)
        input_wubi = input_wubi.masked_fill(~(mask).unsqueeze(-1).unsqueeze(-1), 0)
        input_zhengma = input_zhengma.masked_fill(~(mask).unsqueeze(-1).unsqueeze(-1), 0)

        input_wubi = input_wubi.unsqueeze(2)
        input_zhengma = input_zhengma.unsqueeze(2)

        input = torch.cat([input_wubi, input_zhengma], dim=2)
        batch, max_seq_len, in_channels, max_num, embed_size = input.size()
        input = input.reshape(batch*max_seq_len, 1, in_channels, max_num, embed_size)

        total = []
        for i in range(0, len(Config.kernel_sizes)):
            input1 = F.relu(self.convs[3*i](input)) # conv3d
            input1 = input1.squeeze(2)

            input1 = F.relu(self.convs[3*i+1](input1)) # conv2d
            input1 = torch.max(input1, dim=-1, keepdim=False)[0]  # pool

            input1 = F.relu(self.convs[3*i+2](input1)) # conv1d

            input1 = self.pool(input1).squeeze(-1)  # pool
            input1 = input1.reshape(batch, max_seq_len, -1)
            total.append(input1)
        total = torch.cat(total, dim=-1)
        total = self.hidden2embed(total)
        return total



        # input = self.test1(input) # conv3d
        # input = input.squeeze(2)
        # input = self.test2(input) # conv2d
        # input = torch.max(input, dim=-1, keepdim=False)[0] # pool
        # input = self.test3(input) # conv1d
        # input = self.pool(input).squeeze(-1) # pool
        # input = input.reshape(batch, max_seq_len, -1)
        # print(input.shape)
        # sys.exit()



