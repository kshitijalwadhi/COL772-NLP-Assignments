import sys
import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

classes = ['pH', 'Generic-Measure', 'Speed', 'Temperature', 'Concentration', 'Action', 'Reagent', 'Time', 'Measure-Type', 'Numerical', 'Size', 'Modifier', 'Method', 'Location', 'Mention', 'Amount', 'Device', 'Seal']


def performance_metrics(list_true, list_pred):

    dict_ = {entity: {'tp': 0, 'fn': 0, 'fpo': 0, 'fpd_true': 0, 'fpd_pred': 0, 'tot_fp': 0,
                      'true_entities': 0, 'pred_entities': 0} for entity in set(list_true)|set(list_pred)}


    for (t_entity, p_entity) in zip(list_true, list_pred):
        dict_[t_entity]['true_entities'] += 1
        dict_[p_entity]['pred_entities'] += 1

        # true posititve
        if t_entity == p_entity:
            dict_[t_entity]['tp'] += 1

        # false negative (TRUE <> 'O', PRED = 'O')
        elif (t_entity != 'O') & (p_entity == 'O'):
            dict_[t_entity]['fn'] += 1
            dict_[p_entity]['fn'] += 1

        # false positive (TRUE = 'O', PRED <> 'O')
        elif (t_entity == 'O') & (p_entity != 'O'):
            dict_[p_entity]['fpo'] += 1
            dict_[t_entity]['fpo'] += 1

            dict_[p_entity]['tot_fp'] += 1
            dict_[t_entity]['tot_fp'] += 1

        # false positive (TRUE <> PRED)
        elif (t_entity != p_entity):
            dict_[t_entity]['fpd_true'] += 1
            dict_[p_entity]['fpd_pred'] += 1

            dict_[t_entity]['tot_fp'] += 1

    df = pd.DataFrame(dict_).transpose()

    return df

def create_classification_report(list_true, list_pred):

    df = performance_metrics(list_true, list_pred)                                           # function

    if df.empty:
        classification_report = pd.DataFrame(data=([]),
                                             index=['-'],
                                             columns=['precision', 'recall', 'f1_score', 'true_entities', 'pred_entities'])

        averages_report = pd.DataFrame(data=([]),
                                       index=['micro_avg', 'macro_avg', 'weighted_avg'],
                                       columns=['precision', 'recall', 'f1_score', 'true_entities', 'pred_entities'])

    else:
        # remove the 'O' row if the 'O' tag is present
        df.drop(['O'], inplace=True) if 'O' in df.index else None

        # 1. CREATE CLASSIFICATION REPORT
        # PRECISION - RECALL - F1_SCORE
        df['precision'] = (df['tp'] / df['pred_entities'])
        df['recall'] = (df['tp'] / df['true_entities'])
        df['f1_score'] = ((2 * df['precision'] * df['recall']) /(df['precision'] + df['recall']))

        # Replace NaN values with 0
        df = df.replace(np.nan, 0)

        # CLASSIFICATION REPORT
        classification_report = df[['precision', 'recall', 'f1_score', 'true_entities',
                                    'pred_entities']].sort_values(by=['true_entities'], ascending=False)


        # 2. CREATE AVERAGES REPORT
        # Calculate totals row
        totals = df.sum()

        # Calculate micro averages
        micro_avg_precision = totals['tp'] / totals['pred_entities'] if totals['pred_entities']!=0 else 0
        micro_avg_recall = totals['tp'] / totals['true_entities'] if totals['true_entities']!=0 else 0
        micro_avg_f1 = ((2 * micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)) if (micro_avg_precision + micro_avg_recall)!=0 else 0

        # Calculate macro averages
        macro_avg_precision = totals['precision'] / len(df) if len(df)!=0 else 0
        macro_avg_recall = totals['recall'] / len(df) if len(df)!=0 else 0
        macro_avg_f1 = totals['f1_score'] / len(df) if len(df)!=0 else 0

        # Calculate weighted averages
        weighted_avg_precision = sum((df['precision'] * df['true_entities'])) / totals['true_entities'] if totals['true_entities']!=0 else 0
        weighted_avg_recall = sum((df['recall'] * df['true_entities'])) / totals['true_entities'] if totals['true_entities']!=0 else 0
        weighted_avg_f1 = sum((df['f1_score'] * df['true_entities'])) / totals['true_entities'] if totals['true_entities']!=0 else 0

        # AVERAGES REPORT
        averages_report = pd.DataFrame(data=([
            [micro_avg_precision, micro_avg_recall, micro_avg_f1, totals['true_entities'], totals['pred_entities']],
            [macro_avg_precision, macro_avg_recall, macro_avg_f1, totals['true_entities'], totals['pred_entities']],
            [weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, totals['true_entities'], totals['pred_entities']]
        ]),
            index=['micro_avg', 'macro_avg', 'weighted_avg'],
            columns=['precision', 'recall', 'f1_score', 'true_entities', 'pred_entities'])


        # 3. CONCAT CLASSIFICATION REPORT & AVERAGES REPORT
        classification_report = pd.concat([classification_report, averages_report], axis=0)

    return classification_report

#gold labels
with open(sys.argv[1], "r") as f:
    st = f.read().strip()
    st = re.sub(r'\n\s*\n', '\n\n', st)

lines = st.split("\n\n")
gold = [line.split('\n') for line in lines]
goldlabels = []
for sent in gold:
    sentlabels = [item.split("\t")[1] for item in sent]
    goldlabels += sentlabels
goldlabels = [it.replace("B-","").replace("I-","") for it in goldlabels]
#predictions
with open(sys.argv[2], "r") as f:
    st = f.read().strip()
lines = st.split("\n\n")
predlabels = []
for line in lines:
    predlabels += line.split("\n")

predlabels = [it.replace("B-","").replace("I-","") for it in predlabels]

cr = create_classification_report(goldlabels, predlabels)

print("CLASSIFICATION Report")
print(cr)
print("\nF scores")
print("Micro-F1 = " + str(cr.loc['micro_avg']['f1_score']))
print("Macro-F1 = " + str(cr.loc['macro_avg']['f1_score']))
