from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

from get_samples import get_pre_train_samples


###############################################################################
# Setting up
training_path = '/home/zhihua/work/object_detector/image/training'

X = get_pre_train_samples(training_path, 760, 240, [40, 40])
print('sample is in shape', X.shape)
# Load Data
#digits = datasets.load_digits()
#X = np.asarray(digits.data, 'float32')
#Y = np.asarray(digits.target, 'float32')
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                    test_size=0.2,
#                                                    random_state=0)

# Models we will use
rbm = BernoulliRBM(random_state=0, verbose=True)


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

# Training RBM-Logistic Pipeline
rbm.fit(X)

###############################################################################
# Plotting

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()