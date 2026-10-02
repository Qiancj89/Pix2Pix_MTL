# -*- coding: utf-8 -*-
import sklearn
import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv   
                     
if __name__ == '__main__':
    model_name = ['efficientnet-b7', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']

    matrix = np.zeros((len(model_name),9), dtype='<U32')

    for i in range(0, len(model_name)):
        df1 = pd.read_csv(model_name[i]+'/validation_predictions_hamming_Pix2Pix_enhanced_ceus.csv', usecols=[17])
        label = df1.to_numpy()
        #print(output)
        
        df2 = pd.read_csv(model_name[i]+'/validation_predictions_hamming_Pix2Pix_enhanced_ceus.csv', usecols=[13])
        output = df2.to_numpy()
        #print(output)
        
        loss = hamming_loss(label, output)
        print(f"Hamming Loss: {loss}")

        acc_score = accuracy_score(label, output)
        print(f"Accuracy_Score: {acc_score}")

        pre_score = precision_score(label, output, average='macro')
        print(f"Precision_Score: {pre_score}")

        recall = recall_score(label, output, average='macro')
        print(f"Recall_Score: {recall}")

        f1 = f1_score(label, output, average='macro')
        print(f"macro_F1_Score: {f1}")

        f2 = f1_score(label, output, average='micro')
        print(f"micro_F1_Score: {f2}")

        ROC = roc_auc_score(label, output)
        print(f"AUC_Score: {ROC}")

        ck_score = cohen_kappa_score(label, output)
        print(f"Cohen_Kappa_Score: {ck_score}")

        #cm = confusion_matrix(label, output)
        #print(f"Confusion_Matrix: {cm}")

        matrix[i,0] = model_name[i]
        matrix[i,1] = loss
        matrix[i,2] = acc_score
        matrix[i,3] = pre_score
        matrix[i,4] = recall
        matrix[i,5] = f1
        matrix[i,6] = f2
        matrix[i,7] = ROC
        matrix[i,8] = ck_score

    fname = ['method', 'hamming_loss', 'accuracy', 'precision', 'recall', 
    'macro_F1', 'micro_F1', 'AUC_score', 'Cohen_Kappa_score']
    with open('validation.csv','w',newline='') as fi:
        writer = csv.writer(fi)
        writer.writerow(fname)
        for l in range(0, len(model_name)):
            data_single = matrix[l, :]
            writer.writerow(data_single)
    fi.close()