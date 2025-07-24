#
#       Basic functions to read a csv file, label encoding
#       and column filtering.
#
#       computation of the evaluation metrics (R, P, F1 score, FPR)
#
#       Software is distributed without any warranty;
#       bugs should be reported to antonio.pecchia@unisannio.it
#

import numpy as np
import pandas as pd
from sklearn.preprocessing import (Binarizer, MaxAbsScaler, MinMaxScaler,
                                   Normalizer, RobustScaler, StandardScaler)

# 90 columns ('Label' included)

names = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration',
         'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
         'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
         'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
         'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
         'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
         'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
         'Fwd RST Flags', 'Bwd RST Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
         'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
         'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
         'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
         'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
         'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
         'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
         'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
         'ICMP Code', 'ICMP Type', 'Total TCP Flow Time', 'Label']


#  81 features being used (NOTE: 'Label' is excluded too)
# ( x1,x2,...,xn)

features = ['Flow Duration',
            'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
            'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd RST Flags', 'Bwd RST Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
            'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
            'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
            'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
            'ICMP Code', 'ICMP Type', 'Total TCP Flow Time']


def transformData(training, validation, test):

    dftrain = pd.read_csv(training, names=names, header=None, sep=',', index_col=False, dtype='unicode')
    dfval = pd.read_csv(validation, names=names, header=None, sep=',', index_col=False, dtype='unicode')
    dftest = pd.read_csv(test, names=names, header=None, sep=',', index_col=False, dtype='unicode')

    # invocare getXY #

    X_tr, Y_tr, L_tr = getXY(dftrain)
    X_val, Y_val, L_val = getXY(dfval)
    X_te, Y_te, L_te = getXY(dftest)

    scaler = MaxAbsScaler()

    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    return X_tr, Y_tr, L_tr, X_val, Y_val, L_val, X_te, Y_te, L_te, dftrain, dfval, dftest


def getXY(inDataframe):

    X = inDataframe[features].values.astype(float)

    # 1-hot encoding

    Y0 = np.where(inDataframe['Label'] == 'BENIGN', 1, 0)
    Y1 = np.where(inDataframe['Label'] != 'BENIGN', 1, 0)

    Y = np.column_stack((Y0, Y1))

    # Y = pd.get_dummies(inDataframe['Label']).values.astype(float)

    L = inDataframe['Label'].values

    return X, Y, L


# -----------------------------------------------#
#           instructor-provided code            #
# -----------------------------------------------#
def evaluatePerformance(outcome, evaluationLabels):

    # outcome: boolean      TRUE    ->  BENIGN
    #                       FALSE   ->  ATTACK

    # evaluationLabels:     original labels

    eval = pd.DataFrame(data={'prediction': outcome, 'Class': evaluationLabels})

    TN = 0
    TP = 0
    FN = 0
    FP = 0

    print('')
    print('             *** EVALUATION RESULTS ***')
    print('')
    print('        Coverage by attack (positive classes)')
    classes = eval['Class'].unique()
    # Recall by class
    print('%6s %10s %10s' % ('FN', 'TP', 'recall'))
    for c in classes:
        if c != 'BENIGN':
            A = eval[(eval['prediction'] == True) & (eval['Class'] == c)].shape[0]
            B = eval[(eval['prediction'] == False) & (eval['Class'] == c)].shape[0]

            print('%6d %10d %10.3f %26s' % (A, B, B / (A + B), c))

            FN = FN + A     # cumulative FN
            TP = TP + B     # cumulative TP
        else:
            TN = eval[(eval['prediction'] == True) & (eval['Class'] == 'BENIGN')].shape[0]
            FP = eval[(eval['prediction'] == False) & (eval['Class'] == 'BENIGN')].shape[0]

    print('%6s %10s' % ('----', '----'))
    print('%6d %10d %10s' % (FN, TP, 'total'))

    print('')
    print('Confusion matrix:')

    print('%42s' % ('prediction'))
    print('%36s | %14s' % (' | BENIGN (neg.)', 'ATTACK (pos.)'))
    print('       --------------|---------------|---------------')
    print('%28s  %6d | FP = %9d' % ('BENIGN (neg.) | TN = ', TN, FP))
    print('label  --------------|---------------|---------------')
    print('%28s  %6d | TP = %9d' % ('ATTACK (pos.) | FN = ', FN, TP))
    print('       --------------|---------------|---------------')

    recall = TP / (TP + FN)
    precision = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)
    f1 = 0
    if precision + recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    fpr = FP / (FP + TN)

    print('Metrics:')
    print('R = %5.3f  P = %5.3f  F1 score = %5.3f  FPR = %5.3f' % (recall, precision, f1, fpr))

    return recall, precision, f1, fpr
