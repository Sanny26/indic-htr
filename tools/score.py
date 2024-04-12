import re
import argparse
import numpy as np

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def clean(label):
    alphabet = [a for a in '0123456789abcdefghijklmnopqrstuvwxyz* ']
    label = label.replace('-', '*')
    nlabel = ""
    for each in label.lower():
        if each in alphabet:
            nlabel += each
    return nlabel


if  __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--preds', type=str, default='../misc/preds/temp.txt', help='path to preds file')
	parser.add_argument('--mode', type=str, default='word', choices=['word', 'line'])
	parser.add_argument('--lower', action='store_true', help='convert strings to lowercase ebfore comparison')
	parser.add_argument('--alnum', action='store_true', help='convert strings to alphanumeric before comparison')
	opt = parser.parse_args()

	f = open(opt.preds, 'r')

	tw = 0
	ww = 0
	tc = 0
	wc = 0

	if opt.mode == 'word':
		for i , line in enumerate(f):
			if i%2==0:
				pred = line.strip()
			else:
				gt = line.strip()
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
				tc += len(gt)
				tw += 1
	else:  # logic for line-level comparison
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
