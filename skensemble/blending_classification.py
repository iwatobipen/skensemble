# Blending classifier

# Author: iwatobipen
# License: BSD


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
import numpy as np
import six
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class BlendingClassifier(BaseEstimator, ClassifierMixin):
    """A Blending classifier for scikit-learn estimators for classification.

    Parameters
    ----------
    l1_clfs: array-like, shape = [n_clfs]
        A list of classifiers of first layer.
    l2_clf: object
        The classifier of second layer.
    n_hold: int, optional (default=5)
        number of hold
    test_size: float, optional (default=0.2)
        size of test which is used for second layer training
    verbose: int, optional (default=0)
        Controls the verbosity of the building process.
        - 'verborse=0' (default): prints nothing
        - 'verborse=>1' Prints the number & name of classifer being fitted
    use_clones: bool (default: True)
    random_state (default: 794)

    Attributes
    ----------
    l1_clfs_: list, shape=[n_clfs]
        Fitted l1_classifiers
    l2_clf_: estimator
        Fitted l2_classifier
    num_cls: int (default: None)
        number of classes for classification

    Examples
    ---------
    For usage example, please see
    --comming soon

    """

    def __init__(self, l1_clfs, l2_clf, 
                 n_hold=5, test_size=0.2, verbose=0,
                 use_clones=True, random_state=794):
        self.l1_clfs = l1_clfs
        self.l2_clf = l2_clf

        self.n_hold = n_hold
        self.test_size = test_size
        self.verbose = verbose
        self.use_clones = use_clones
        self.random_state = random_state
        self.num_cls = None

    def fit(self, X, y):
        """Fit ensemble l1_clfs and l2_clf.
        X: array-like, shape = [n_samples, n_features]
            Traingin vectors
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values
        """
        skf = StratifiedKFold(n_splits=self.n_hold, random_state=self.random_state)
        if self.use_clones:
            self.l1_clfs_ = [clone(clf) for clf in self.l1_clfs]
            print(len(self.l1_clfs_))
            self.l2_clf_ = clone(self.l2_clf)
        else:
            self.l1_clfs_ = self.l1_clfs
            self.l2_clf_ = self.l2_clf

        self.num_cls = len(set(y))
        if self.verbose > 0:
            print("Fitting {} l1_classifiers...".format(len(self.l1_clfs)))
            print("{} classes classification".format(self.num_cls))
        
        dataset_blend_train = np.zeros((X.shape[0],len(self.l1_clfs_), self.num_cls))

        for j, clf in enumerate(self.l1_clfs_):
            for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                if self.verbose > 0:
                    print('{}-{}th hold, {} classifier'.format(j+1, i+1, type(clf)))
                train_i_x, train_i_y  = X[train_idx], y[train_idx]
                test_i_x, test_i_y = X[test_idx], y[test_idx]
                clf.fit(train_i_x, train_i_y)
                dataset_blend_train[test_idx, j, :] = clf.predict_proba(test_i_x)
 
        if self.verbose > 0:
            print('--- Blending ---')
            print(dataset_blend_train.shape)
        
        dataset_blend_train = dataset_blend_train.reshape((dataset_blend_train.shape[0], -1))
        self.l2_clf_.fit(dataset_blend_train, y)
        return self


    def predict(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X: array-like shape = [n_sample, n_features]
            Training vectors

        Returns
        -------
        labels: array-like, shape = [n_samples] or [n_samples, n_outputs]
            Predicted class labels:
        """

        l1_output = np.zeros((X.shape[0], len(self.l1_clfs_), self.num_cls))
        for i, clf in enumerate(self.l1_clfs_):
            pred_y = clf.predict_proba(X)
            l1_output[:, i, :] = pred_y
        l1_output = l1_output.reshape((X.shape[0], -1))
        return self.l2_clf_.predict(l1_output)


    def predict_proba(self, X):
        """Predict target values for X.

        Parameters
        ----------
        X: array-like shape = [n_sample, n_features]
            Training vectors

        Returns
        -------
        labels: array-like, shape = [n_samples, num_cls]
            Predicted class labels:
        """
        l1_output = np.zeros((X.shape[0], len(self.l1_clfs_), self.num_cls))
        for i, clf in enumerate(self.l1_clfs_):
            pred_y = clf.predict_proba(X)
            l1_output[:, i, :] = pred_y
        l1_output = l1_output.reshape((X.shape[0], -1))
        return self.l2_clf_.predict_proba(l1_output)
