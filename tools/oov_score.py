import re
import argparse
from score import clean, levenshtein


parser = argparse.ArgumentParser()
parser.add_argument('--preds', type=str, required=True, help='file containing training data word list')
parser.add_argument('--tr_vocab', type=str, required=True, help='path to preds file')
parser.add_argument('--mode', type=str, default='word', choices=['word', 'line'])
parser.add_argument('--lower', action='store_true', help='convert strings to lowercase before comparison')
parser.add_argument('--alnum', action='store_true', help='convert strings to alphanumeric before comparison')
opt = parser.parse_args()

train_vocab = []
with open(opt.tr_vocab) as f:
    for line in f:
        train_vocab.append(line.strip())


f = open(opt.preds, 'r')

tw = 0
ww = 0
tc = 0
wc = 0

word_lens = []
if opt.mode == 'word':
    for i , line in enumerate(f):
        if i%2==0:
            pred = line.strip()
        else:
            gt = line.strip()
            if gt in train_vocab:
                continue
            if opt.lower:
                gt = gt.lower()
                pred = pred.lower()
            if opt.alnum:
                pattern = re.compile('[\W_]+')
                gt = pattern.sub('', gt)
                pred = pattern.sub('', pred)
            if gt != pred:
                ww += 1
                wc += levenshtein(gt, pred)
                word_lens.append(len(gt))
            tc += len(gt)
            tw += 1
else:
    for i , line in enumerate(f):
        if i%2==0:
            pred = line.strip()
        else:
            gt = line.strip()
            gt = clean(gt)
            pred = clean(pred)
            gt_w = gt.split()
            pred_w = pred.split()
            for j in range(len(gt_w)):
                try:
                    if gt_w[j] != pred_w[j]:
                        # print(gt_w[j], pred_w[j])
                        ww += 1
                except IndexError:
                    ww += 1

            tw += len(gt.split())
            wc += levenshtein(gt, pred)
            tc += len(gt)


print(ww, tw)
print('WER: ', (ww/tw)*100)
print('CER: ', (wc/tc)*100)
print('Incorrect Avg: ', sum(word_lens)/len(word_lens))
print('Incorrect Max Avg: ', max(word_lens))
print('Incorrect Min Avg: ', min(word_lens))
