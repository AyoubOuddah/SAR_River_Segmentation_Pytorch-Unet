import numpy as np

def separteConfMatrix(confMat):
    TP = confMat[1, 1]
    TN = confMat[0, 0]
    FP = confMat[0, 1]
    FN = confMat[1, 0]
    return TP, TN, FP, FN

def accuracy(confMat):
    TP, TN, FP, FN = separteConfMatrix(confMat)
    return (TP + TN) / (TP + TN + FP + FN)

def recall(confMat):
    TP, TN, FP, FN = separteConfMatrix(confMat)
    return TP / (TP + FN)

def precision(confMat):
    TP, TN, FP, FN = separteConfMatrix(confMat)
    return TP / (TP + FP)

def scoreF1(confMat):
    pre = precision(confMat)
    rec = recall(confMat)
    return (2 * pre * rec) / (pre + rec)
