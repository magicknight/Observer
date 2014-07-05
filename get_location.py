__author__ = 'zhihua'

from sklearn.cluster import DBSCAN
import numpy as np


# This function use DBSCAN clustering to select the center that most likely as a lesion
def get_location(predict_positions, target_size):
    db = DBSCAN(eps=target_size/2, min_samples=2).fit(predict_positions)
    labels = db.labels_
    print 'labels are', labels
    cluster_labels = np.delete(labels, np.argwhere(labels == -1))
    if cluster_labels.size == 0:
        return [-1, -1], 1
    print 'after deleted -1, labels are', cluster_labels
    # find cluster that has most points
    label = np.argmax(np.bincount(cluster_labels.astype(np.int64)))
    print 'most frequency label is', label
    class_members = [index[0] for index in np.argwhere(labels == label)]
    print 'class members are', class_members
    positions = [predict_positions[i] for i in class_members]
    print 'the final picked positions are:', positions
    return np.mean(positions, axis=0, dtype=np.int64), len(class_members)