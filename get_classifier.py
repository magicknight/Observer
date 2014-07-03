__author__ = 'zhihua'
from sklearn import svm
from sklearn.linear_model import SGDClassifier

# definitions
#########################################################


def get_classifier(name):
    """

    :param name:
    :param training_sample:
    :param training_label:
    :return:classifier object
    """
    if name=='svm':
        # training SVM and dump the trained svm to a binary file
        clf = svm.SVC(class_weight='auto', cache_size=3000, C=1.0)
        return clf

    if name=='sgd':
        # training SVM and dump the trained svm to a binary file
        clf = SGDClassifier(loss="hinge", penalty="l2")
        return clf