import sys
from load_data import idx2label, traindataloader
from models import Bert_lattice, Bert_lattice_for_wubi
from fastNLP.embeddings import BertEmbedding
from fastNLP import seq_len_to_mask
from utils import get_crf_zero_init, MyDropout
from read_embedding import vocabs, embeddings
import torch.nn as nn
import torch
from Config import Config
from torch.nn.utils.rnn import pad_sequence
from wubi_embedding import TextCNN

device = "cuda" if torch.cuda.is_available() else 'cpu'
bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', requires_grad=True,
                                           word_dropout=0.01, include_cls_sep=False)
WUBI = {}
wubi2idx = {}
wubi2idx['z'] = len(wubi2idx)
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
ZHENGMA = {}
zhengma2idx = {}
zhengma2idx['zz'] = len(zhengma2idx)
with open(r'/xiaowang/ner_dataset/shiyan1/zhengma.txt', encoding='utf-8', mode='r') as f:
    ff = f.readlines()
    for idx, pair in enumerate(ff):
        pair = pair.strip('\n').split(' ')
        if pair[0] not in ZHENGMA:
            ZHENGMA[pair[0]]= pair[1]
        for word in pair[1]:
            if word not in zhengma2idx:
                zhengma2idx[word] = len(zhengma2idx)
def get_wubi(raw_chars):
    total = []
    for chars in raw_chars:
        sen = []
        for word in chars:
            words = []
            if word in WUBI:
                for i in WUBI[word]:
                    words.append(wubi2idx[i])
                if len(WUBI[word]) < 4:
                    for _ in range(4 - len(WUBI[word])):
                        words.append(wubi2idx['z'])
            else:
                for _ in range(4):
                    words.append(wubi2idx['z'])
            words = torch.tensor(words)
            sen.append(words)
        sen = pad_sequence(sen, batch_first=True, padding_value=0)
        total.append(sen)
    total = pad_sequence(total, padding_value=0, batch_first=True)
    total = total.to(device)
    return total
def get_zhengma(raw_chars):
    total = []
    for chars in raw_chars:
        sen = []
        for word in chars:
            words = []
            if word in ZHENGMA:
                for i in ZHENGMA[word]:
                    words.append(zhengma2idx[i])
                if len(ZHENGMA[word]) < 4:
                    for _ in range(4 - len(ZHENGMA[word])):
                        words.append(zhengma2idx['zz'])
            else:
                for _ in range(4):
                    words.append(zhengma2idx['zz'])
            words = torch.tensor(words)
            sen.append(words)
        sen = pad_sequence(sen, batch_first=True, padding_value=0)
        total.append(sen)
    total = pad_sequence(total, batch_first=True, padding_value=0)
    total = total.to(device)
    return total
class Bert_Ner(nn.Module):
    def __init__(self, idx2label, use_bigrams, use_bert, use_wubi, use_MyDropout = Config.use_MyDropout):
        super(Bert_Ner, self).__init__()
        self.bert = bert_embedding
        self.label_size = len(idx2label)
        self.crf = get_crf_zero_init(self.label_size, include_start_end_trans=False)
        self.hidden_size = Config.total
        self.use_bigrams = use_bigrams
        self.use_bert = use_bert
        self.use_wubi = use_wubi
        self.use_MyDropout = use_MyDropout

        if self.use_MyDropout:
            self.dropout_for_output = MyDropout(Config.dropout_for_output)
            self.dropout_for_char_embedding = MyDropout(Config.dropout_for_char_embedding)
            self.dropout_for_lex_embeddding = MyDropout(Config.dropout_for_lex_embedding)
        else:
            self.dropout_for_output = nn.Dropout(Config.dropout_for_output)
            self.dropout_for_char_embedding = nn.Dropout(Config.dropout_for_char_embedding)
            self.dropout_for_lex_embeddding = nn.Dropout(Config.dropout_for_lex_embedding)
        if self.use_wubi:
            self.attn_for_wubi = Bert_lattice_for_wubi()
        self.attn = Bert_lattice()
        self.attn_for_total = Bert_lattice_for_wubi()

        self.lattice_embedding = embeddings['lattice']
        self.bigrams_embedding = embeddings['bigram']
        if self.use_wubi:
            self.wubi_embedding = TextCNN(embed_size=Config.embed_size, kernel_sizes=Config.kernel_sizes,
                                      num_channels=Config.num_channels)
        char_input_size = self.lattice_embedding.embedding.weight.size(1)
        char_input_size_for_wubi = 0
        if self.use_bigrams:
            char_input_size += self.bigrams_embedding.embedding.weight.size(1)
            char_input_size_for_wubi += self.bigrams_embedding.embedding.weight.size(1)
        if self.use_bert:
            char_input_size += self.bert._embed_size
            char_input_size_for_wubi += self.bert._embed_size
        if self.use_wubi:
            char_input_size_for_wubi += self.wubi_embedding.hidden_size

        self.char_hidden2attn = nn.Linear(char_input_size, self.hidden_size)
        self.char_hidden2attn_for_wubi = nn.Linear(char_input_size_for_wubi, self.hidden_size)
        self.lex_hidden2attn = nn.Linear(self.lattice_embedding.embedding.weight.size(1), self.hidden_size)
        # if self.use_wubi:
        #     self.hidden2label = nn.Linear(self.hidden_size * 2, self.label_size)
        # else:
        #     self.hidden2label = nn.Linear(self.hidden_size, self.label_size)
        self.hidden2label = nn.Linear(self.hidden_size, self.label_size)
        #self.hidden_two_label = nn.Linear(self.hidden_size*2, self.label_size)


    def forward(self, input, mode='train'):
        batch = input['lattice'].size(0)
        max_seq_len = input['bigrams'].size(1)
        max_seq_len_and_lex = input['lattice'].size(1)

        lattice_embedding_raw = self.lattice_embedding(input['lattice'])
        raw_char = lattice_embedding_raw

        wubi = get_wubi(input['raw_chars'])
        zhengma = get_zhengma(input['raw_chars'])

        wubi_embedding = self.wubi_embedding(wubi, zhengma, input['seq_len'])
        mask_for_char_for_wubi = seq_len_to_mask(torch.tensor(input['seq_len']))
        mask_for_char_for_wubi = mask_for_char_for_wubi.to(device)
        total_for_wubi = wubi_embedding

        if self.use_bigrams:
            bigrams_embedding = self.bigrams_embedding(input['bigrams'])
            bigrams_embedding1 = torch.cat([bigrams_embedding,
                                            torch.zeros(size=[batch, max_seq_len_and_lex - max_seq_len,
                                                              bigrams_embedding.size(2)],
                                                        device=device)], dim=1)
            raw_char = torch.cat([raw_char, bigrams_embedding1], dim=-1)
            total_for_wubi = torch.cat([total_for_wubi, bigrams_embedding], dim=-1)

        if self.use_bert:
            words_id = input['input_ids']
            words_id = words_id.to(device)
            bert_emdedding = self.bert(words_id)
            bert_emdedding1 = torch.cat([bert_emdedding,
                                         torch.zeros(size=[batch, max_seq_len_and_lex - max_seq_len,
                                                           bert_emdedding.size(2)],
                                                     device=device)], dim=1)
            raw_char = torch.cat([bert_emdedding1, raw_char], dim=-1)
            total_for_wubi = torch.cat([total_for_wubi, bert_emdedding], dim=-1)
        char_embedding = self.char_hidden2attn(raw_char)
        mask_for_lattice = seq_len_to_mask(torch.tensor(input['seq_len']), max_len=max_seq_len_and_lex)
        mask_for_lattice = mask_for_lattice.to(device)
        char_embedding = char_embedding.masked_fill(~(mask_for_lattice.unsqueeze(-1)), 0)
        char_embedding = self.dropout_for_char_embedding(char_embedding)

        raw_lex = lattice_embedding_raw
        lex_embedding = self.lex_hidden2attn(raw_lex)
        char_and_lex_num = torch.tensor([i + j for i, j in zip(input['seq_len'], input['lex_num'])])
        mask_for_lex = seq_len_to_mask(char_and_lex_num, max_len=max_seq_len_and_lex)
        mask_for_lex = mask_for_lex.to(device)
        lex_embedding = lex_embedding.masked_fill(~(mask_for_lex.unsqueeze(-1)), 0)
        lex_embedding = self.dropout_for_lex_embeddding(lex_embedding)

        emb_total = char_embedding + lex_embedding

        encoded = self.attn(input1=emb_total, input2=emb_total, input3=emb_total,
                            lex_num=input['lex_num'], real_lengths=input['seq_len'],
                            pos_s=input['pos_s'], pos_e=input['pos_e'])

        encoded = encoded[:, :max_seq_len, :]

        total = self.char_hidden2attn_for_wubi(total_for_wubi)
        total = total.masked_fill(~(mask_for_char_for_wubi.unsqueeze(-1)), 0)
        total = self.dropout_for_char_embedding(total)

        encoded_wubi = self.attn_for_wubi(total, total, total, input['lex_num'], input['seq_len'],
                                          pos_s=input['pos_s'][:, :max_seq_len],
                                          pos_e=input['pos_e'][:, :max_seq_len])


        encoded = self.attn_for_total(encoded_wubi, encoded, encoded, input['lex_num'], input['seq_len'],
                                              pos_s=input['pos_s'][:, :max_seq_len],
                                              pos_e=input['pos_e'][:, :max_seq_len])
        #pred = self.hidden_two_label(torch.cat([encoded, encoded_wubi], dim=-1))

        pred = self.hidden2label(encoded)
        pred = self.dropout_for_output(pred)

        mask = seq_len_to_mask(torch.tensor(input['seq_len'])).bool()
        mask = mask.to(device)

        if mode == 'train':
            loss = self.crf(pred, input['target'], mask).mean(dim=0)
            return loss

        elif mode == 'test':
            pred, path = self.crf.viterbi_decode(pred, mask)
            return pred




# model = Bert_Ner(idx2label=idx2label, use_bert=Config.use_bert, use_bigrams=Config.use_bigrams, use_wubi=Config.use_wubi)
# model.to(device)
# for i in traindataloader:
#     model(i)
#     sys.exit()