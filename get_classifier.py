__author__ = 'zhihua'
from sklearn import svm

# definitions
#########################################################


def get_classifier(name, training_sample, training_label):
    """

    :param name:
    :param training_sample:
    :param training_label:
    :return:classifier object
    """
    if name=='svm':
        # training SVM and dump the trained svm to a binary file
        clf = svm.SVC(class_weight='auto', cache_size=3000, C=1000)
        clf.fit(training_sample, training_label)
        return clf
