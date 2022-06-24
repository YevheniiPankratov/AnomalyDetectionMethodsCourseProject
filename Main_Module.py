from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances as dist_matrix
from scipy.linalg import qr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import base
from sklearn import utils
from warnings import warn
from scipy.stats import scoreatpercentile
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
from pylab import rcParams

rnd = np.random.RandomState()

class LOF(base.NeighborsBase, base.KNeighborsMixin, base.UnsupervisedMixin):
    def __init__(self, n_neighbors=20, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1):
        self._init_params(n_neighbors=n_neighbors,
                          algorithm=algorithm,
                          leaf_size=leaf_size, metric=metric, p=p,
                          metric_params=metric_params, n_jobs=n_jobs)

        self.contamination = contamination

    def fit_predict(self, X, y=None):
        return self.fit(X)._predict()

    def fit(self, X, y=None):
        if not (0. < self.contamination <= .5):
            raise ValueError("contamination must be in (0, 0.5]")

        super(LOF, self).fit(X)

        n_samples = self._fit_X.shape[0]
        if self.n_neighbors > n_samples:
            warn("n_neighbors (%s) is greater than the "
                 "total number of samples (%s). n_neighbors "
                 "will be set to (n_samples - 1) for estimation."
                 % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self._distances_fit_X_, _neighbors_indices_fit_X_ = (
            self.kneighbors(None, n_neighbors=self.n_neighbors_))

        self._lrd = self._local_reachability_density(
            self._distances_fit_X_, _neighbors_indices_fit_X_)

        # Обчислюйте оцінку lof за навчальними зразками, щоб визначити поріг_:
        lrd_ratios_array = (self._lrd[_neighbors_indices_fit_X_] /
                            self._lrd[:, np.newaxis])

        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        self.threshold_ = -scoreatpercentile(
            -self.negative_outlier_factor_, 100. * (1. - self.contamination))

        return self

    def _predict(self, X=None):
        if X is not None:
            X = utils.check_array(X, accept_sparse='csr')
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self._decision_function(X) <= self.threshold_] = -1
        else:
            is_inlier = np.ones(self._fit_X.shape[0], dtype=int)
            is_inlier[self.negative_outlier_factor_ <= self.threshold_] = -1

        return is_inlier

    def decision_function(self, X):
        X = utils.check_array(X, accept_sparse='csr')

        distances_X, neighbors_indices_X = (
            self.kneighbors(X, n_neighbors=self.n_neighbors_))
        X_lrd = self._local_reachability_density(distances_X,
                                                 neighbors_indices_X)

        lrd_ratios_array = (self._lrd[neighbors_indices_X] /
                            X_lrd[:, np.newaxis])

        # чим більше, тим краще:
        return -np.mean(lrd_ratios_array, axis=1)

    def _local_reachability_density(self, distances_X, neighbors_indices):
        dist_k = self._distances_fit_X_[neighbors_indices,
                                        self.n_neighbors_ - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)

        # 1e-10, щоб уникнути `nan', коли немає дублікатів > n_neighbors:
        return 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)

class PolynomSolver(BaseEstimator):
    def __init__(self, metric='chebyshev'):
        self.metric = metric

    def fit(self, X):
        self.X = X
        return self

    def decision_function(self, X):
        DM = dist_matrix(X, self.X, metric=self.metric)
        ans = (DM ** (1 / len(self.X))).prod(axis=1)
        return -ans

class RndWeight(BaseEstimator):
    def __init__(self, BaseAlgorithm=IsolationForest(1)):
        self.BaseAlgorithm = copy.deepcopy(BaseAlgorithm)

    def apply_weights(self, D_):
        D = D_.copy()
        for i in range(D.shape[1]):
            D[:, i] *= self.weights[i]
        return D
    def fit(self, X):
        self.weights = rnd.uniform(0, 1, X.shape[1])
        self.BaseAlgorithm.fit(self.apply_weights(X))
        return self
    def decision_function(self, X):
        return self.BaseAlgorithm.decision_function(self.apply_weights(X))

class Rotated(BaseEstimator):  # будуємо багато модифікованих iTree і усереднюємо
    def __init__(self, BaseAlgorithm=IsolationForest(1)):
        self.BaseAlgorithm = copy.deepcopy(BaseAlgorithm)

    def rotate(self, X):  # просто модифікуємо всі основні функції алгоритма, поставлена замість визнаних пов'язаних
        return X.dot(self.RotationMatrix)

    def fit(self, X):
        # генеруємо випадкову матрицю повороту
        H = rnd.uniform(0, 1, (X.shape[1], X.shape[1]))
        self.RotationMatrix, R = qr(H)
        self.BaseAlgorithm.fit(self.rotate(X))
        return self
    def decision_function(self, X):
        return self.BaseAlgorithm.decision_function(self.rotate(X))

class Ensembler(BaseEstimator):
    def __init__(self, BaseAlgorithm=IsolationForest(), n_estimators=100):
        self.BaseAlgorithm = BaseAlgorithm
        self.n_estimators = n_estimators

    def fit(self, X):
        self.algorithms = []
        for i in range(self.n_estimators):
            self.algorithms.append(copy.deepcopy(self.BaseAlgorithm).fit(X))
        return self

    def decision_function(self, New_X):
        ans = np.zeros((len(New_X)))
        for i in range(self.n_estimators):
            ans += self.algorithms[i].decision_function(New_X)
        return ans / self.n_estimators

class Sampler(BaseEstimator):
    def __init__(self, BaseAlgorithm=IsolationForest(), n_estimators=100, max_samples=250):
        self.BaseAlgorithm = BaseAlgorithm
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X):
        self.algorithms = []
        for i in range(self.n_estimators):
            indexes = rnd.choice(len(X), self.max_samples)
            self.algorithms.append(copy.deepcopy(self.BaseAlgorithm).fit(X[indexes]))
        return self

    def decision_function(self, New_X):
        ans = np.zeros((len(New_X)))
        for i in range(self.n_estimators):
            ans += self.algorithms[i].decision_function(New_X)
        return ans / self.n_estimators

class IterativeEnsemble(BaseEstimator):
    def __init__(self, thr = 50, metric="manhattan", estimators=1000):
        self.metric = metric
        self.thr = thr
        self.estimators = estimators

    def fit(self, X):
        pred = IsolationForest(self.estimators, max_samples=250).fit(X).decision_function(X)
        self.model = Sampler(PolynomSolver(metric=self.metric), self.estimators).fit(X[pred > np.percentile(pred, self.thr)])
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)

# Інші функціі

def anomalyScoresPlot(normal, anomaly):
    rcParams['figure.figsize'] = 16, 3
    plt.scatter(normal, np.zeros_like(normal), marker='|', c='green', s=8000)
    plt.scatter(anomaly, np.zeros_like(anomaly), marker='|', c='red', s=8000)
    plt.tick_params(axis='y', left='off', labelleft='off')
    plt.xlim(min(normal.min(), anomaly.min()), max(normal.max(), anomaly.max()))
    plt.show()

def coolPlot(name, decision_func, train=np.empty((0, 2)), test=np.empty((0, 2)), outliers=np.empty((0, 2))):
    rcParams['figure.figsize'] = 14, 12
    f, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[6, 1]})
        
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = decision_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax0.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    b1 = ax0.scatter(train[:, 0], train[:, 1], c='white')
    b2 = ax0.scatter(test[:, 0], test[:, 1], c='green')
    c = ax0.scatter(outliers[:, 0], outliers[:, 1], c='red')
    ax0.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
    ax0.axis('tight')
    ax0.set_xlim((-5, 5))
    ax0.set_ylim((-5, 5))
       
    b = ax1.scatter(np.zeros_like(test), test, marker='_', c='green', s=2000)
    c = ax1.scatter(np.zeros_like(outliers), outliers, marker='_', c='red', s=2000)
    ax1.set_xticks(np.array([]))
    ax1.yaxis.tick_right()
    ax1.set_ylim((outliers.min(), test.max()))
    
    plt.tight_layout()
    st = f.suptitle(name, fontsize="x-large")
    f.subplots_adjust(top=0.95)
    plt.show()


# In[6]:

def aucroc(y_pred_test, y_pred_outliers):
    return roc_auc_score(np.append(np.ones((len(y_pred_test))), -np.ones((len(y_pred_outliers)))), np.append(y_pred_test, y_pred_outliers))

def Solve(clf, train, train_ans, test, test_ans):
    clf.fit(train)
    y_pred_train = clf.decision_function(train[~train_ans])
    y_pred_train_outliers = clf.decision_function(train[train_ans])
    y_pred_test = clf.decision_function(test[~test_ans])
    y_pred_outliers = clf.decision_function(test[test_ans])

    return aucroc(y_pred_train, y_pred_train_outliers), aucroc(y_pred_test, y_pred_outliers)

# In[7]:

def f_measure(y, f, verbose=True):
    right_shots = np.logical_and(f == 1, f == y).sum()
    precision = right_shots / (f == 1).sum()
    recall = right_shots / (y == 1).sum()
    if verbose:
        print("precision = ", precision)
        print("recall = ", recall)
        print("f-measure = ", precision * recall / (0.5 * precision + 0.5 * recall))
    return precision * recall / (0.5 * precision + 0.5 * recall)

def potential_f_measure(normaly_score, anomaly_score):
    rcParams['figure.figsize'] = 16, 3
    f_meas = np.array([])
    for i in np.sort(anomaly_score):
        f_meas = np.append(f_meas, f_measure(np.append(np.zeros_like(normaly_score), np.ones_like(anomaly_score)), 
                                             np.append(normaly_score <= i, anomaly_score <= i), verbose=False))
                      
    print("Potential max f-measure = ", f_meas.max(), ' (thr = ', np.sort(anomaly_score)[f_meas.argmax()], ')')
    all_points = np.append(normaly_score, anomaly_score)
    plt.xlim(all_points.min(), all_points.max())
    plt.plot(np.sort(anomaly_score), f_meas)