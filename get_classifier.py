__author__ = 'zhihua'
import cPickle as pickle
from sklearn import svm
import os

# definitions
svm_file = 'svm.pkl'
#########################################################


def get_classifier(name, training_sample, training_label):
    """

    :param name:
    :param training_sample:
    :param training_label:
    :return:classifier object
    """
    if name=='svm':
        #load SVM if there exist trained SVM file.
        if os.path.isfile(svm_file):
            with open(svm_file, 'rb') as fid:
                clf = pickle.load(fid)
        else:
            #if no svm file exist, train it
            # training SVM and dump the trained svm to a binary file
            clf = svm.SVC(class_weight='auto', cache_size=3000)
            clf.fit(training_sample, training_label)
            with open(svm_file, 'wb') as fid:
                pickle.dump(clf, fid)
        return clf