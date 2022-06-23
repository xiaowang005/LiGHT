import sys
from collections import Counter
from models1 import Bert_Ner
from load_data import testdataloader, idx2label, label2idx, devdataloader, traindataloader
import torch
from Config import Config
device = "cuda" if torch.cuda.is_available() else 'cpu'
model = Bert_Ner(label2idx, use_bigrams=Config.use_bigrams, use_bert=Config.use_bert, use_wubi=Config.use_wubi)
model_path = '/xiaowang/ner_dataset/shiyan1/Flat-Wubi/saved_model/model5'
states = torch.load(model_path).state_dict()
model.load_state_dict(states)
model.to(device)
model.eval()
use_print = False

def extract(chars, tags):
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

            # if idx == len(tags)-1:
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

gold_num = 0
pred_num = 0
correct_num = 0
preds = []
reals = []
count = 1
true_entity = []
for idx, batch_data in enumerate(testdataloader):
    if idx%100 ==0:
        print(idx)
    pred = model(batch_data, mode='test')[0]
    target = batch_data['target'][0]

    raw_chars = batch_data['raw_chars'][0]
    #sentence = ['S'] + raw_chars + ['E']
    sentence = raw_chars

    true_labels = [idx2label[i.item()] for i in target]
    true_entities = extract(sentence, true_labels)
    new_true_entities = true_entities[:len(true_entities)]
    true_entity.extend(new_true_entities)
    gold_num += len(new_true_entities)

    pred_labels = [idx2label[i.item()] for i in pred]
    pred_entities = extract(sentence, pred_labels)
    new_pred_entities = pred_entities[:len(pred_entities)]
    pred_num += len(new_pred_entities)

    preds.extend(pred_labels)
    reals.extend(true_labels)

    if use_print:
        print('Sents: ', sentence, count)
        print('NER: ', new_true_entities)
        print('Pred_NER: ', new_pred_entities)
        print('\n')
    count += 1

    for i in new_pred_entities:
        if i in new_true_entities:
            correct_num += 1

pre = correct_num / pred_num
rec = correct_num / gold_num
f1 = 2*pre*rec/(pre+rec)
# print('gold_num: ', gold_num)
# print('pred_num: ', pred_num)
# print('correct_num: ', correct_num)
# print('Pre: ', pre)
# print('Rec: ', rec)
# print('F1: ', f1)
# print('\n')


def flatten_lists(lists):
    flatten_lists = []
    for list_ in lists:
        if type(list_) == list:
            flatten_lists.extend(list_)
        else:
            flatten_lists.append(list_)
    return flatten_lists

class Metrics(object):
    """评价模型，计算每个标签的精确率、召回率、F1分数"""
    def __init__(self,gloden_tags,predict_tags,remove_0=False):
        self.golden_tags = flatten_lists(gloden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_0:   # 不统计非实体标记
            self._remove_Otags()

        # 所有的tag总数
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        # print(self.correct_tags_number)
        self.predict_tags_count = Counter(self.predict_tags)
        self.golden_tags_count = Counter(self.golden_tags)

        # 精确率
        self.precision_scores = self.cal_precision()
        # 召回率
        self.recall_scores = self.cal_recall()
        # F1
        self.f1_scores = self.cal_f1()

    def cal_precision(self):
        """计算每个标签的精确率"""
        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = 0 if self.correct_tags_number.get(tag,0)==0 else \
                self.correct_tags_number.get(tag,0) / self.predict_tags_count[tag]

        return precision_scores

    def cal_recall(self):
        """计算每个标签的召回率"""
        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag,0) / self.golden_tags_count[tag]

        return recall_scores

    def cal_f1(self):
        """计算f1分数"""
        f1_scores = {}
        for tag in self.tagset:
            f1_scores[tag] = 2*self.precision_scores[tag]*self.recall_scores[tag] / \
                                    (self.precision_scores[tag] + self.recall_scores[tag] + 1e-10)
        return f1_scores

    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_scores(self,dtype='HMM'):
        """将结果用表格的形式打印出来，像这个样子：

                      precision    recall  f1-score   support
              B-LOC      0.775     0.757     0.766      1084
              I-LOC      0.601     0.631     0.616       325
             B-MISC      0.698     0.499     0.582       339
             I-MISC      0.644     0.567     0.603       557
              B-ORG      0.795     0.801     0.798      1400
              I-ORG      0.831     0.773     0.801      1104
              B-PER      0.812     0.876     0.843       735
              I-PER      0.873     0.931     0.901       634

          avg/total      0.779     0.764     0.770      6178
        """
        # 打印表头
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        with open('/xiaowang/lstm-crf-NER/doc.test/new_code/result.txt',mode='a') as fout:
            fout.write('\n')
            fout.write('=========='*10)
            fout.write('\n')
            fout.write('模型：{}，test结果如下：'.format(dtype))
            fout.write('\n')
            fout.write(header_format.format('', *header))
            print(header_format.format('', *header))

            row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
            # 打印每个标签的 精确率、召回率、f1分数
            for tag in self.tagset:
                print(row_format.format(
                    tag,
                    self.precision_scores[tag],
                    self.recall_scores[tag],
                    self.f1_scores[tag],
                    self.golden_tags_count[tag]
                ))
                fout.write('\n')
                fout.write(row_format.format(
                    tag,
                    self.precision_scores[tag],
                    self.recall_scores[tag],
                    self.f1_scores[tag],
                    self.golden_tags_count[tag]
                ))

            # 计算并打印平均值
            avg_metrics = self._cal_weighted_average()
            print(row_format.format(
                'avg/total',
                avg_metrics['precision'],
                avg_metrics['recall'],
                avg_metrics['f1_score'],
                len(self.golden_tags)
            ))
            fout.write('\n')
            fout.write(row_format.format(
                'avg/total',
                avg_metrics['precision'],
                avg_metrics['recall'],
                avg_metrics['f1_score'],
                len(self.golden_tags)
            ))
            fout.write('\n')


    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_count[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))

sss = Counter(preds)
print(sss)
print('\n')
# metrics = Metrics(reals, preds, remove_0=True)
# dtype = 'Bi_LSTM+CRF'
# metrics.report_scores(dtype=dtype)

print('gold_num: ', gold_num)
print('pred_num: ', pred_num)
print('correct_num: ', correct_num)
print('Pre: ', pre)
print('Rec: ', rec)
print('F1: ', f1)
print('\n')


