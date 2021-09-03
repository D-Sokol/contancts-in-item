#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('gt', type=pd.read_csv, help="Файл со входными данными и столбцом is_bad")
parser.add_argument('pred', type=pd.read_csv, help="Выходной файл программы")
args = parser.parse_args()

gt = args.gt
pr = args.pred

scores = []
for category in gt['category'].unique():
    y_true = gt[gt['category'] == category]['is_bad']
    y_pred = pr[gt['category'] == category]['prediction']
    score = roc_auc_score(y_true, y_pred)
    print(category, score, sep='\t')
    scores.append(score)

print()

print("Среднее значение:", np.mean(scores), sep='\t')
