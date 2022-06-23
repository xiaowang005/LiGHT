import copy
import math
import sys

from Config import Config as Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
device = "cuda" if torch.cuda.is_available() else 'cpu'
def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb

class Get_four_embedding(nn.Module):
    def __init__(self, pe, pe_ss, pe_se, pe_es, pe_ee, mode=Config.four_pos_mode, hidden = Config.total):
        super(Get_four_embedding, self).__init__()
        self.mode = mode
        self.pe = pe
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee

        if self.mode == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(2*hidden, hidden),
                                                    nn.ReLU(inplace=True))
        else:
            self.pos_fusion_forward = nn.Sequential(nn.Linear(4*hidden, hidden),
                                                    nn.ReLU(inplace=True))

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)
        max_seq_len = pos_s.size(1)

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(1)
        pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(1)
        pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(1)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(1)

        pos_ss = self.pe_ss[pos_ss.view(-1)+Config.max_sep_len].view(batch, max_seq_len, max_seq_len, Config.total)
        pos_se = self.pe_se[pos_se.view(-1)+Config.max_sep_len].view(batch, max_seq_len, max_seq_len, Config.total)
        pos_es = self.pe_es[pos_es.view(-1)+Config.max_sep_len].view(batch, max_seq_len, max_seq_len, Config.total)
        pos_ee = self.pe_ee[pos_ee.view(-1)+Config.max_sep_len].view(batch, max_seq_len, max_seq_len, Config.total)

        if self.mode == 'ff_two':
            result = torch.cat([pos_ss, pos_ee], dim=-1)
            result = result.to(device)
            result = self.pos_fusion_forward(result)
        else:
            result = torch.cat([pos_ss, pos_se, pos_es, pos_ee], dim=-1)
            result = result.to(device)
            result = self.pos_fusion_forward(result)

        return result

class Multi_head_attention(nn.Module):
    def __init__(self, num_head, hidden_size, q_proj, k_proj, v_proj, r_proj):
        super(Multi_head_attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.w_q = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.w_k = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.w_v = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.w_r = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.u = nn.Parameter(torch.Tensor(self.num_head, self.hidden_size), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(self.num_head, self.hidden_size), requires_grad=True)

    def forward(self, query, key, value, lex_num, real_lengths, rel_pos_embedding):
        input = query
        if self.q_proj:
            query = self.w_q(query)
        if self.k_proj:
            key = self.w_k(key)
        if self.v_proj:
            value = self.w_v(value)
        if self.r_proj:
            rel_pos_embedding = self.w_r(rel_pos_embedding)
        batch = query.size(0)
        max_seq_len = rel_pos_embedding.size(1)

        #query = query.view(batch, -1, self.num_head, self.hidden_size)
        # key = key.view(batch, -1, self.num_head, self.hidden_size)
        # value = value.view(batch, -1, self.num_head, self.hidden_size)
        # rel_pos_embedding = rel_pos_embedding.view(batch, max_seq_len, max_seq_len, self.num_head, self.hidden_size)
        query = torch.reshape(query, [batch, -1, self.num_head, self.hidden_size])
        key = torch.reshape(key, [batch, -1, self.num_head, self.hidden_size])
        value = torch.reshape(value, [batch, -1, self.num_head, self.hidden_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding, [batch, max_seq_len, max_seq_len, self.num_head, self.hidden_size])

        ## 注意力头到前面
        query = query.transpose(1, 2) ## batch, num_head, seq_len, hidden
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        ## 相对位置矩阵注意力头到前面 # batch, num_head, seq_len, hidden, seq_len
        rel_pos_embedding = rel_pos_embedding.permute(0, 3, 1, 4, 2)

        key = key.transpose(-1, -2) ## batch, num_head, hidden, seq_len
        #u_for_c = self.u.view(1, self.num_head, 1, self.hidden_size)
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        query_and_u_for_c = query + u_for_c
        # batch, head,
        A_C = torch.matmul(query_and_u_for_c, key)

        query_for_b = query.view(batch, self.num_head, max_seq_len, 1, self.hidden_size)
        v_for_d = self.v.view(1, self.num_head, 1, 1, self.hidden_size)
        query_for_b_and_v_for_d = query_for_b + v_for_d
        ## batch, head, seq_len, hidden
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding).squeeze(-2)

        attn_score = A_C + B_D

        attn_score_mask = seq_len_to_mask(torch.tensor([i for i in real_lengths])+torch.tensor(lex_num))
        attn_score_mask = attn_score_mask.to(device)
        attn_score = attn_score.masked_fill(~(attn_score_mask.unsqueeze(1).bool().unsqueeze(1)), -1e15)

        attn_score = F.softmax(attn_score, dim=-1) # batch, num_head, seq_len, seq_len
        if attn_score.size(3) > 40:
            #attn_score = F.softmax(attn_score, dim=-1)  # batch, num_head, seq_len, seq_len
            attn_score = attn_score.squeeze(0)
            # print(attn_score.shape)
            # print(attn_score[5])
            attn = torch.zeros(size=[attn_score.size(2), attn_score.size(2)])
            attn = attn.to(device)
            for i in attn_score:
                attn += i
            #print(attn)
            ss = nn.Sigmoid()
            #attn = ss(attn)
            print(attn)

            pic_add = attn.cpu().detach().numpy()

            import numpy as np;
            np.random.seed(0)
            import seaborn as sns;
            sns.set()
            import matplotlib.pyplot as plt
            uniform_data = np.random.rand(24, 24)
            f, ax = plt.subplots(figsize=(9, 6))
            ax = sns.heatmap(pic_add, square=True)
            plt.show()
            # for i in range(8):
            #     ax = sns.heatmap(attn_score[i], square=True)
            #     plt.show()
            sys.exit()


        # if attn_score.size(3) > 100:
        #     attn_score = attn_score.cpu().detach().numpy()
        #     import matplotlib.pyplot as plt
        #     import matplotlib as mpl
        #     zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/liberation/SimHei.ttf',
        #                                              size=12)  # 给zhfont添加中文黑体的属性
        #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        #
        #     plt.rcParams['xtick.direction'] = 'in'
        #     plt.rcParams['ytick.direction'] = 'in'
        #
        #     import seaborn
        #     rc = {'font.sans-serif': 'SimHei',
        #           'axes.unicode_minus': False}
        #     seaborn.set(context='talk', style='ticks', rc=rc)
        #
        #     seaborn.set(font='SimHei')
        #
        #     # seaborn.set_context(context="talk")
        #
        #     def draw(data, ax):
        #         seaborn.heatmap(data,
        #                         square=True, vmin=0.0, vmax=1.0,
        #                         cbar=False, ax=ax)
        #
        #     fig, axs = plt.subplots(1, 8, figsize=(40, 20))  # 布置画板
        #     plt.xticks(fontproperties=zhfont)
        #     plt.yticks(fontproperties=zhfont)
        #     for h in range(8):  # 每一个循环都是一个头的注意力
        #         plt.xticks(fontproperties=zhfont)  # 这两句是指定画图中要把坐标轴定义为中文
        #         plt.yticks(fontproperties=zhfont)
        #         draw(attn_score[h].squeeze(0), ax=axs[h])
        #     plt.show()
        #     sys.exit()


        final_result = torch.matmul(attn_score, value) ## batch, num_head, seq_len, hidden

        final_result = final_result.transpose(1, 2).contiguous().\
            view(batch, max_seq_len, self.num_head * self.hidden_size)

        return nn.LayerNorm(Config.total).cuda()(input+final_result)

class Positionwise_Feedforward(nn.Module):
    def __init__(self, num_head, hidden_size, ff = Config.ff, attn_active = Config.active, add = Config.add_ff):
        super(Positionwise_Feedforward, self).__init__()
        self.ff = ff
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.total = self.num_head * self.hidden_size
        self.add = add
        self.dropout1 = nn.Dropout(Config.dropout_for_ff_1)
        self.dropout2 = nn.Dropout(Config.dropout_for_ff_2)

        if attn_active == 'Relu':
            self.active = nn.ReLU(inplace=True)
        self.position_forward = nn.Sequential(nn.Linear(self.total, self.ff),
                                              self.active,
                                              self.dropout1,
                                              nn.Linear(self.ff, self.total),
                                              self.active,
                                              self.dropout2)
        self.linear1 = nn.Linear(self.total, self.ff)
        self.linear2 = nn.Linear(self.ff, self.total)
    def forward(self, input):
        inp = input
        input = self.position_forward(input)

        if self.add:
            input = input + inp
        return nn.LayerNorm(self.total).cuda()(input)


class Transformer_layer(nn.Module):
    def __init__(self, num_head, hidden_size, q_proj, k_proj, v_proj, r_proj):
        super(Transformer_layer, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.attn = Multi_head_attention(self.num_head, self.hidden_size, self.q_proj, self.k_proj, self.v_proj, self.r_proj)
        self.ff = Positionwise_Feedforward(self.num_head, self.hidden_size)
        self.layer_process  = Layer_process(self.num_head, self.hidden_size)
    def forward(self, input1, input2, input3, lex_num, real_lengths, rel_pos_embedding):
        output = self.attn(input1, input2, input3, lex_num, real_lengths, rel_pos_embedding)
        #output = self.layer_process(output)
        output = self.ff(output)
        return output

class Layer_process(nn.Module):
    def __init__(self, num_head, hidden_size, add=Config.add_layer):
        super(Layer_process, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.d_model = self.hidden_size * self.num_head
        self.add = add
    def forward(self, input):
        inp = input
        if self.add:
            input = input + inp
        return nn.LayerNorm(self.d_model).cuda()(input)

class Bert_lattice(nn.Module):
    def __init__(self, q_proj=Config.q_proj, k_proj=Config.k_proj, v_proj=Config.v_proj, r_proj=Config.r_proj,
                 num_layers = Config.num_layers, num_head = Config.num_head, hidden_size = Config.hidden_size):
        super(Bert_lattice, self).__init__()
        self.pe = get_embedding(Config.max_sep_len, Config.total, rel_pos_init=0)
        self.four_pos_shared = Config.four_pos_shared
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.num_layers = num_layers
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dev = device
        self.pe = nn.Parameter(self.pe, requires_grad=True)

        if self.four_pos_shared:
            self.pe_ss = self.pe
            self.pe_se = self.pe
            self.pe_es = self.pe
            self.pe_ee = self.pe
        else:
            self.pe_ss = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
            self.pe_se = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
            self.pe_es = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
            self.pe_ee = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
        self.get_rel_pos_embedding = Get_four_embedding(self.pe, self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                                    mode=Config.four_pos_mode, hidden=Config.total)
        self.get_rel_pos_embedding = self.get_rel_pos_embedding.to(self.dev)
        self.layers = nn.ModuleList([Transformer_layer(self.num_head, self.hidden_size,
                                                       self.q_proj, self.k_proj, self.v_proj, self.v_proj) for _ in range(self.num_layers)])

    def forward(self, input1, input2, input3, lex_num, real_lengths, pos_s, pos_e):
        rel_pos_embedding = self.get_rel_pos_embedding(pos_s, pos_e)
        for layer in self.layers:
            input = layer(input1, input2, input3, lex_num, real_lengths, rel_pos_embedding)

        return input



class Bert_lattice_for_wubi(nn.Module):
    def __init__(self, q_proj=Config.q_proj, k_proj=Config.k_proj, v_proj=Config.v_proj, r_proj=Config.r_proj,
                 num_layers = Config.num_layers, num_head = Config.num_head, hidden_size = Config.hidden_size):
        super(Bert_lattice_for_wubi, self).__init__()
        self.pe = get_embedding(Config.max_sep_len, Config.total, rel_pos_init=0)
        self.four_pos_shared = Config.four_pos_shared
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.num_layers = num_layers
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dev = device
        self.pe = nn.Parameter(self.pe, requires_grad=True)

        if self.four_pos_shared:
            self.pe_ss = self.pe
            self.pe_se = self.pe
            self.pe_es = self.pe
            self.pe_ee = self.pe
        else:
            self.pe_ss = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
            self.pe_se = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
            self.pe_es = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
            self.pe_ee = nn.Parameter(copy.deepcopy(self.pe), requires_grad=True)
        self.get_rel_pos_embedding = Get_four_embedding(self.pe, self.pe_ss, self.pe_se, self.pe_es, self.pe_ee,
                                                    mode=Config.four_pos_mode, hidden=Config.total)
        self.get_rel_pos_embedding = self.get_rel_pos_embedding.to(self.dev)
        self.layers = nn.ModuleList([Transformer_layer_for_wubi(self.num_head, self.hidden_size,
                                                       self.q_proj, self.k_proj, self.v_proj, self.v_proj) for _ in range(self.num_layers)])

    def forward(self, input1, input2, input3, lex_num, real_lengths, pos_s, pos_e):
        rel_pos_embedding = self.get_rel_pos_embedding(pos_s, pos_e)
        for layer in self.layers:
            input = layer(input1, input2, input3, lex_num, real_lengths, rel_pos_embedding)

        return input

class Transformer_layer_for_wubi(nn.Module):
    def __init__(self, num_head, hidden_size, q_proj, k_proj, v_proj, r_proj):
        super(Transformer_layer_for_wubi, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj
        self.attn = Multi_head_attention_for_wubi(self.num_head, self.hidden_size, self.q_proj, self.k_proj, self.v_proj, self.r_proj)
        self.ff = Positionwise_Feedforward(self.num_head, self.hidden_size)
        self.layer_process  = Layer_process(self.num_head, self.hidden_size)
    def forward(self, input1, input2, input3, lex_num, real_lengths, rel_pos_embedding):
        output = self.attn(input1, input2, input3, lex_num, real_lengths, rel_pos_embedding)
        output = self.ff(output)
        return output

class Multi_head_attention_for_wubi(nn.Module):
    def __init__(self, num_head, hidden_size, q_proj, k_proj, v_proj, r_proj):
        super(Multi_head_attention_for_wubi, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.w_q = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.w_k = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.w_v = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.w_r = nn.Linear(self.hidden_size * self.num_head, self.hidden_size * self.num_head)
        self.u = nn.Parameter(torch.Tensor(self.num_head, self.hidden_size), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(self.num_head, self.hidden_size), requires_grad=True)

    def forward(self, query, key, value, lex_num, real_lengths, rel_pos_embedding):
        input = query
        if self.q_proj:
            query = self.w_q(query)
        if self.k_proj:
            key = self.w_k(key)
        if self.v_proj:
            value = self.w_v(value)
        if self.r_proj:
            rel_pos_embedding = self.w_r(rel_pos_embedding)
        batch = query.size(0)
        max_seq_len = rel_pos_embedding.size(1)

        #query = query.view(batch, -1, self.num_head, self.hidden_size)
        # key = key.view(batch, -1, self.num_head, self.hidden_size)
        # value = value.view(batch, -1, self.num_head, self.hidden_size)
        # rel_pos_embedding = rel_pos_embedding.view(batch, max_seq_len, max_seq_len, self.num_head, self.hidden_size)
        query = torch.reshape(query, [batch, -1, self.num_head, self.hidden_size])
        key = torch.reshape(key, [batch, -1, self.num_head, self.hidden_size])
        value = torch.reshape(value, [batch, -1, self.num_head, self.hidden_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding, [batch, max_seq_len, max_seq_len, self.num_head, self.hidden_size])

        ## 注意力头到前面
        query = query.transpose(1, 2) ## batch, num_head, seq_len, hidden
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        ## 相对位置矩阵注意力头到前面 # batch, num_head, seq_len, hidden, seq_len
        rel_pos_embedding = rel_pos_embedding.permute(0, 3, 1, 4, 2)

        key = key.transpose(-1, -2) ## batch, num_head, hidden, seq_len
        #u_for_c = self.u.view(1, self.num_head, 1, self.hidden_size)
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        query_and_u_for_c = query + u_for_c
        # batch, head,
        A_C = torch.matmul(query_and_u_for_c, key)

        query_for_b = query.view(batch, self.num_head, max_seq_len, 1, self.hidden_size)
        v_for_d = self.v.view(1, self.num_head, 1, 1, self.hidden_size)
        query_for_b_and_v_for_d = query_for_b + v_for_d
        ## batch, head, seq_len, hidden
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding).squeeze(-2)

        attn_score = A_C + B_D

        attn_score_mask = seq_len_to_mask(torch.tensor([i for i in real_lengths]))
        attn_score_mask = attn_score_mask.to(device)
        attn_score = attn_score.masked_fill(~(attn_score_mask.unsqueeze(1).unsqueeze(1)), -1e15)

        attn_score = F.softmax(attn_score, dim=-1) # batch, num_head, seq_len, seq_len

        final_result = torch.matmul(attn_score, value) ## batch, num_head, seq_len, hidden

        final_result = final_result.transpose(1, 2).contiguous().\
            view(batch, max_seq_len, self.num_head * self.hidden_size)

        return nn.LayerNorm(Config.total).cuda()(input+final_result)