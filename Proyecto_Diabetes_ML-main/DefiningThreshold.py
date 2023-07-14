import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve
from yellowbrick.classifier import ClassPredictionError
from sklearn.exceptions import NotFittedError

class ThresholdOptimizer:
    """Class that receives a trained model, the data and the labels output. It returns the standard metrics (with threshold = 0.5), threshold vs metrics graphs 
    
    and recommended threshold points and 
    
    """
    def __init__(self):
        self._metrics_list = None
        self._model = None
        self._data = None
        self._labels = None
        self._prob = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.accuracy = None
        self.auc = None             
        self._max_precision = None
        self._max_recall = None
        self._precision_recall_curve_cut = None
        self._max_auc = None
        self._max_f1 = None
        self._max_accuracy = None
        
    def fit(self, model, data, labels, metrics_list = None):
        self._metrics_list = metrics_list
        self._model = model
        self._data = data
        self._labels = labels
        self._prob = cross_val_predict(self._model, self._data, self._labels, cv=5, method='predict_proba') 
     
        if self._metrics_list is not None:
            #Get precision and recall values for every threshold value
            self.precision, self.recall, self.thresholds = precision_recall_curve(self._labels, self._prob[:,1])
            self._max_precision = self.thresholds[np.argmax(self.precision)]
            self._max_recall = self.thresholds[np.argmax(self.recall)] 
            self._precision_recall_curve_cut = self.__cut(self.thresholds, self.precision, self.recall).mean()
            #Add the last datapoint to the threshold list
            self.thresholds = np.append(self.thresholds, 1)
            #Get f1,auc and accuracy values for every threshold value.
            for i in self._metrics_list:
                if i != "precision" and i != "recall":
                    if i == "f1":
                        self.f1 = np.array([metrics.f1_score(self._labels, self.__adjust_classes(self._prob[:,1], t=j)) for j in self.thresholds])
                        self._max_f1 = self.thresholds[np.argmax(self.f1)]
                    elif i == "accuracy":
                        self.accuracy = np.array([metrics.accuracy_score(self._labels, self.__adjust_classes(self._prob[:,1], t=j)) for j in self.thresholds])
                        self._max_accuracy = self.thresholds[np.argmax(self.accuracy)] 
                    elif i == "auc":
                        self.auc = np.array([roc_auc_score(self._labels, self.__adjust_classes(self._prob[:,1], t=j )) for j in self.thresholds])
                        self._max_auc = self.thresholds[np.argmax(self.auc)]
                    else:
                        raise ValueError("Invalid metric.")   
        return self
    
    def predict(self, new_data, metric):
        try:
            if metric == "precision":
                thr = self._max_precision
            elif metric == "recall":
                thr = self._max_recall
            elif metric == "auc":
                thr = self._max_auc
            elif metric == "f1":
                thr = self._max_f1
            elif metric == "acccuracy":
                thr = self._max_accuracy
            elif metric == "precision_recall_curves_cut":
                thr = self._precision_recall_curve_cut
            return np.array([1 if y >= thr else 0 for y in self._model.predict_proba(new_data)[:,1]])
        except ValueError:
            print(f'Metric {metric} is not defined.')
        except NotFittedError:
            print("Optimizer not fitted, call fit before exploiting the model.")

    def predict_proba(self, new_data, metric):
        return self._model.predict_proba(new_data)

    def get_default_metrics(self):
        #Metrics
        scoring = {
        'accuracy': 'accuracy',
        'precision':'precision',
        'recall': 'recall',
        'f1' : 'f1'
        }

        scores= cross_validate(self._model,self._data,self._labels,scoring = scoring)
        print("Default model metrics are:")
        print("Accuracy: {:.3f} +- {:.3f}".format(scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
        print("F1: {:.3f} +- {:.3f}".format(scores['test_f1'].mean(), scores['test_f1'].std()))
        print("Precision: {:.3f} +- {:.3f}".format(scores['test_precision'].mean(),scores['test_precision'].std()))
        print("Recall: {:.3f} +- {:.3f}".format(scores['test_recall'].mean(),scores['test_recall'].std()),"\n")
        
        # Confusion matrix
        # Get a list of classes that matches the columns of `prob`
        sorted_prob = np.unique(self._labels)
        # Use the highest probability for predicting the label
        indices = np.argmax(self._prob, axis=1)
        # Get the label for each sample
        pred = sorted_prob[indices]
        cm= pd.DataFrame(confusion_matrix(self._labels, pred),
                            columns=['pred_neg', 'pred_pos'], 
                            index=['neg', 'pos'])

        print("Default confusion matrix:", "\n", cm,"\n" )
        
    def __cut(self,thresholds, precision, recall):
        num = []
        for pos, val in enumerate(thresholds):
            if abs(precision[pos] - recall[pos]) <= 0.001:
                num.append(val)
        return np.array(num)
    
    def __adjust_classes(self,probas, t):
        return np.array([1 if y >= t else 0 for y in probas])
                   
    def plot_metrics(self):
        #Plot     
        if self._metrics_list is not None:
            if "precision" in self._metrics_list:
                plt.plot(self.thresholds, self.precision, color=sns.color_palette()[0], label = "Precision")
            if "recall" in self._metrics_list:
                  plt.plot(self.thresholds, self.recall, color=sns.color_palette()[1], label = "Recall")              
            if "auc" in self._metrics_list:
                plt.plot(self.thresholds, self.auc,  color=sns.color_palette()[2], label = "AUCS")
            if "f1" in self._metrics_list:
                plt.plot(self.thresholds, self.f1,  color=sns.color_palette()[3], label = "F1")
            if "accuracy" in self._metrics_list:
                plt.plot(self.thresholds, self.accuracy,  color=sns.color_palette()[4], label = "Accuracy")
    
        plt.xlabel("Threshold")
        plt.ylabel("Metrics")
        plt.legend()
        plt.show()
        
    def get_metrics_maximums(self): 
        if "precision" in self._metrics_list:
            print("Precision is maximized at th ={:.3f}".format(self._max_precision))
        if "recall" in self._metrics_list:    
            print("Recall is maximized at th ={:.3f}".format(self._max_recall))
        if "precision" in self._metrics_list and "recall" in self._metrics_list:
            print("The recall and precision curves intersect at th = {:.3f}".format(self._precision_recall_curve_cut))         
        if "auc" in self._metrics_list:
            print("AUC is maximized at th ={:.3f}".format(self._max_auc))
        if "f1" in self._metrics_list:    
            print("F1 is maximized at th ={:.3f}".format(self._max_f1))
        if "accuracy" in self._metrics_list:   
            print("Accuracy is maximized at th ={:.3f}".format(self._max_accuracy))
             
    def change_threshold(self, t):
        adjusted_labels = self.__adjust_classes(self._prob[:,1], t)
        cm= pd.DataFrame(confusion_matrix(self._labels, adjusted_labels),
                                columns=['pred_neg', 'pred_pos'], 
                                index=['neg', 'pos'])
            
        print("\n","th = {}".format(t),"\n", cm)
        print("Accuracy: {:.3f}".format(metrics.accuracy_score(self._labels, adjusted_labels)))
        print("Precision: {:.3f}".format(metrics.precision_score(self._labels, adjusted_labels)))
        print("Recall: {:.3f}".format(metrics.recall_score(self._labels,adjusted_labels)))
        print("F1: {:.3f}".format(metrics.f1_score(self._labels,adjusted_labels)))
        print("AUC: {:.3f}".format(metrics.roc_auc_score(self._labels,adjusted_labels)))
        print("----------------------------")
