from sklearn.metrics import (accuracy_score, precision_score, 
                             confusion_matrix, cohen_kappa_score, 
                             matthews_corrcoef, f1_score,
                             roc_auc_score, recall_score, balanced_accuracy_score)
import numpy as np
"""
Metricas de evaluacion a usar:

Accuracy - Listo
Accuracy per class - Listo
Class Accuracy - Listo
Kappa - Listo
Mathews Correlation Coeficient - Listo
F1-Score - Listo
mAUC (Area bajo la curva)
AUC
Precision - Listo
Recall - Listo

"""

class MericsEvaluation():
    def __init__(self, pred, labels, sets):

        assert len(pred) == len(labels), "No equal size predictions and labels"
        self.preds = pred
        self.labels = labels
        self.set = sets

    def dataSize(self):
        elements_repeats = set(self.labels)
        repeats = {}
        for elemento in elements_repeats:
            repeats[elemento] = self.labels.count(elemento)
        return tuple(repeats.items())
    
    def accuracy(self):
        return accuracy_score(self.labels, self.preds)
    
    def accuracy_per_class(self):
        preds = confusion_matrix(self.labels, self.preds).diagonal().tolist()
        spc = self.dataSize()
        return tuple([preds[i] / spc[i][1] for i in range(len(spc))])
    
    def accuracy02(self):
        return ((self.accuracy_per_class()[0] + self.accuracy_per_class()[2]) / 2)

    def accuracy134(self):
        return ((self.accuracy_per_class()[1] + self.accuracy_per_class()[3] + self.accuracy_per_class()[4]) / 3)
    
    def class_accuracy(self):
        return sum(self.accuracy_per_class()) / len(self.accuracy_per_class())
    
    def kappa(self):
        return cohen_kappa_score(self.labels, self.preds, weights='quadratic')

    def matthews_coeficent(self):
        return matthews_corrcoef(self.labels, self.preds)
    
    def balanced_accuracy(self):
        return balanced_accuracy_score(self.labels, self.preds)

    def f1(self):
        return f1_score(self.labels, self.preds, average='macro')
    
    def mAUC(self):
        return roc_auc_score(self.labels, self.preds,multi_class='ovr')
    
    def precision(self):
        return precision_score(self.labels, self.preds, average='macro')
    
    def recall(self):
        return recall_score(self.labels, self.preds, average='macro')
    
    def getall(self):
        a, b, c, d, e = self.accuracy_per_class()
        result = [
            self.accuracy(), self.class_accuracy(), self.accuracy02(), self.accuracy134(), a, b, c, d, e,
            self.kappa(), self.matthews_coeficent(), self.f1(), self.precision(), self.recall() 
        ]

        #print(confusion_matrix(self.labels, self.preds).tolist())
        return result

# metrics = MericsEvaluation([1,1,1,2,3,3,3,3,4,4,4,0,0,0,0,0],[2,1,2,2,3,3,3,4,4,4,4,0,0,0,0,0], 'ddr')
# print(metrics.accuracy())
# print(metrics.accuracy_per_class())
# print(metrics.class_accuracy())
# print(metrics.kappa())
# print(metrics.matthews_coeficent())
# print(metrics.f1())
# print(metrics.precision())
# print(metrics.recall())

    
