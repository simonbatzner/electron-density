"""" ML mapping from external potential to charge density - AP275 Class Project, Harvard University

References:
    [1] Brockherde et al. Bypassing the Kohn-Sham equations with machine learning. Nature Communications 8, 872 (2017)
    [2] KRR work based on http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-plot-kernel-ridge-regression-py (Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>)

Simon Batzner, Steven Torrisi, Jon Vandermause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

# KRR
model = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                     param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                 "gamma": np.logspace(-2, 2, 5)})

model.fit(x_trainval, y_trainval)
y_predict_test = model.predict(x_test)

# plot
plt.scatter(x_trainval, y_trainval, label='training data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_test, y_predict_test)
plt.xlabel('data')
plt.ylabel('labels')
plt.title('Kernel Ridge Regression')
plt.legend()
