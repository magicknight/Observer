from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from get_samples import get_pre_train_samples
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from get_samples import get_hog_samples
from sklearn.pipeline import Pipeline

###############################################################################
# Setting up
training_path = '/home/zhihua/work/object_detector/image/training'
#define the parameters
dim_x = 760
dim_y = 195
dim_z = 240
orientations = 9
target_size = 40
pixels_per_cell = (4, 4)
cells_per_block = (1, 1)  # not ready to change this value
scan_window_size = (target_size/pixels_per_cell[0], target_size/pixels_per_cell[1])  # on pixels

print('get training set')
training_sample, training_label, dummy = get_hog_samples(training_path, dim_x, dim_z, orientations, pixels_per_cell,
                                                         cells_per_block, scan_window_size, print_image=False,
                                                         training=True)
print('Training set contains', len(training_label), 'samples')

#training_sample = get_pre_train_samples(training_path, 760, 240, [40, 40])
# Models we will use
X_train, X_test, Y_train, Y_test = train_test_split(training_sample, training_label,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0
# Training RBM-Logistic Pipeline
# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier.predict(X_test))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))
###############################################################################
# Plotting

plt.figure(figsize=(12.6, 12))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((10, 10)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()