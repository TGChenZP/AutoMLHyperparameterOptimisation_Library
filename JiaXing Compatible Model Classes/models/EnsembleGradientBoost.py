import copy
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import numpy as np


class Ensemble_ExplainableBoostingRegressor:

    def __init__(self,
                 max_bins,
                 min_sample_leaf,
                 interactions,
                 max_leaves,
                 learning_rate,
                 max_rounds,
                 n_jobs,
                 max_interaction_bins,
                 ensemble_n_estimators=100,
                 ensemble_max_features=1.0,
                 ensemble_max_samples=1.0,
                 ensemble_weighted=False,
                 **kwargs):

        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins
        self.min_sample_leaf = min_sample_leaf
        self.interactions = interactions
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.n_jobs = n_jobs
        self.max_interaction_bins = max_interaction_bins

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

        self._initialise()

    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(
            10000000, size=self.ensemble_n_estimators)

        for i in range(self.ensemble_n_estimators):

            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(
                train_x, train_y, i)

            ebr = ExplainableBoostingRegressor(max_bins=self.max_bins,
                                               min_sample_leaf=self.min_sample_leaf,
                                               interactions=self.interactions,
                                               max_leaves=self.max_leaves,
                                               learning_rate=self.learning_rate,
                                               max_rounds=self.max_rounds,
                                               max_interaction_bins=self.max_interaction_bins,
                                               n_jobs=self.n_jobs)

            ebr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(ebr)
            self.regressors_train_score.append(
                ebr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [
            x/sum(self.regressors_train_score) for x in self.regressors_train_score]

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(
            train_data, random_state=self.seeds[nth], train_size=self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(
            sampled_train_x.columns), random_state=self.seeds[nth], train_size=self.ensemble_max_features)

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):

            curr_pred = self.regressors[i].predict(
                x[self.regressors_x_columns[i]])

            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i]
                               * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j] /
                               self.ensemble_n_estimators for j in range(len(curr_pred))]

        return predictions


class Ensemble_CatBoostRegressor:

    def __init__(self,
                 n_estimators,
                 max_depth,
                 subsample,
                 colsample_bylevel,
                 max_bin,
                 reg_lambda,
                 learning_rate,
                 random_state,
                 ensemble_n_estimators=100,
                 ensemble_max_features=1.0,
                 ensemble_max_samples=1.0,
                 ensemble_weighted=False,
                 **kwargs):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

        self._initialise()

    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(
            10000000, size=self.ensemble_n_estimators)

        for i in range(self.ensemble_n_estimators):

            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(
                train_x, train_y, i)

            cbr = CatBoostRegressor(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    subsample=self.subsample,
                                    colsample_bylevel=self.colsample_bylevel,
                                    max_bin=self.max_bin,
                                    reg_lambda=self.reg_lambda,
                                    learning_rate=self.learning_rate,
                                    random_state=self.random_state,
                                    verbose=False)

            cbr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(cbr)
            self.regressors_train_score.append(
                cbr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [
            x/sum(self.regressors_train_score) for x in self.regressors_train_score]

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(
            train_data, random_state=self.seeds[nth], train_size=self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(
            sampled_train_x.columns), random_state=self.seeds[nth], train_size=self.ensemble_max_features)

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):

            curr_pred = self.regressors[i].predict(
                x[self.regressors_x_columns[i]])

            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i]
                               * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j] /
                               self.ensemble_n_estimators for j in range(len(curr_pred))]

        return predictions


class Ensemble_LGBMRegressor:

    def __init__(self,
                 n_estimators,
                 max_depth,
                 subsample,
                 colsample_bytree,
                 learning_rate,
                 reg_alpha,
                 n_jobs,
                 random_state,
                 ensemble_n_estimators=100,
                 ensemble_max_features=1.0,
                 ensemble_max_samples=1.0,
                 ensemble_weighted=False,
                 **kwargs):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.learning_rate = learning_rate

        self.n_jobs = n_jobs
        self.random_state = random_state

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

        self._initialise()

    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(
            10000000, size=self.ensemble_n_estimators)

        for i in range(self.ensemble_n_estimators):

            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(
                train_x, train_y, i)

            lgbr = LGBMRegressor(n_estimators=self.n_estimators,
                                 max_depth=self.max_depth,
                                 subsample=self.subsample,
                                 colsample_bytree=self.colsample_bytree,
                                 reg_alpha=self.reg_alpha,
                                 learning_rate=self.learning_rate,
                                 n_jobs=self.n_jobs,
                                 random_state=self.random_state)

            lgbr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(lgbr)
            self.regressors_train_score.append(
                lgbr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [
            x/sum(self.regressors_train_score) for x in self.regressors_train_score]

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(
            train_data, random_state=self.seeds[nth], train_size=self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(
            sampled_train_x.columns), random_state=self.seeds[nth], train_size=self.ensemble_max_features)

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):

            curr_pred = self.regressors[i].predict(
                x[self.regressors_x_columns[i]])

            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i]
                               * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j] /
                               self.ensemble_n_estimators for j in range(len(curr_pred))]

        return predictions


class Ensemble_XGBRegressor:

    def __init__(self,
                 n_estimators,
                 max_depth,
                 subsample,
                 colsample_bytree,
                 reg_alpha,
                 gamma,
                 eta,
                 n_jobs,
                 random_state,
                 ensemble_n_estimators=100,
                 ensemble_max_features=1.0,
                 ensemble_max_samples=1.0,
                 ensemble_weighted=False,
                 **kwargs):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.eta = eta
        self.reg_alpha = reg_alpha

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

        self._initialise()

    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(
            10000000, size=self.ensemble_n_estimators)

        for i in range(self.ensemble_n_estimators):

            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(
                train_x, train_y, i)

            xgbr = XGBRegressor(n_estimators=self.n_estimators,
                                max_depth=self.max_depth,
                                subsample=self.subsample,
                                colsample_bytree=self.colsample_bytree,
                                gamma=self.gamma,
                                eta=self.eta,
                                reg_alpha=self.reg_alpha,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)

            xgbr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(xgbr)
            self.regressors_train_score.append(
                xgbr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [
            x/sum(self.regressors_train_score) for x in self.regressors_train_score]

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(
            train_data, random_state=self.seeds[nth], train_size=self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(
            sampled_train_x.columns), random_state=self.seeds[nth], train_size=self.ensemble_max_features)

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):

            curr_pred = self.regressors[i].predict(
                x[self.regressors_x_columns[i]])

            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i]
                               * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j] /
                               self.ensemble_n_estimators for j in range(len(curr_pred))]

        return predictions


class Ensemble_GradientBoostingRegressor:

    def __init__(self,
                 n_estimators,
                 max_depth,
                 subsample,
                 max_features,
                 ccp_alpha,
                 learning_rate,
                 random_state,
                 ensemble_n_estimators=100,
                 ensemble_max_features=1.0,
                 ensemble_max_samples=1.0,
                 ensemble_weighted=False,
                 **kwargs):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

        self._initialise()

    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(
            10000000, size=self.ensemble_n_estimators)

        for i in range(self.ensemble_n_estimators):

            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(
                train_x, train_y, i)

            gbr = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            subsample=self.subsample,
                                            max_features=self.max_features,
                                            ccp_alpha=self.ccp_alpha,
                                            learning_rate=self.learning_rate,
                                            random_state=self.random_state)

            gbr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(gbr)
            self.regressors_train_score.append(
                gbr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [
            x/sum(self.regressors_train_score) for x in self.regressors_train_score]

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(
            train_data, random_state=self.seeds[nth], train_size=self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(
            sampled_train_x.columns), random_state=self.seeds[nth], train_size=self.ensemble_max_features)

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):

            curr_pred = self.regressors[i].predict(
                x[self.regressors_x_columns[i]])

            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i]
                               * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j] /
                               self.ensemble_n_estimators for j in range(len(curr_pred))]

        return predictions


class Ensemble_HistGradientBoostingRegressor:

    def __init__(self,
                 max_depth,
                 max_bins,
                 interaction_cst,
                 learning_rate,
                 l2_regularization,
                 random_state,
                 ensemble_n_estimators=100,
                 ensemble_max_features=1.0,
                 ensemble_max_samples=1.0,
                 ensemble_weighted=False,
                 **kwargs):

        self.max_depth = max_depth
        self.max_bins = max_bins
        self.interaction_cst = interaction_cst
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

        self._initialise()

    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(
            10000000, size=self.ensemble_n_estimators)

        for i in range(self.ensemble_n_estimators):

            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(
                train_x, train_y, i)

            hgbr = HistGradientBoostingRegressor(HistGradientBoostingRegressor(max_depth=self.max_depth,
                                                                               max_bins=self.max_bins,
                                                                               interaction_cst=self.interaction_cst,
                                                                               learning_rate=self.learning_rate,
                                                                               l2_regularization=self.l2_regularization,
                                                                               random_state=self.random_state))

            hgbr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(hgbr)
            self.regressors_train_score.append(
                hgbr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [
            x/sum(self.regressors_train_score) for x in self.regressors_train_score]

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(
            train_data, random_state=self.seeds[nth], train_size=self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(
            sampled_train_x.columns), random_state=self.seeds[nth], train_size=self.ensemble_max_features)

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):

            curr_pred = self.regressors[i].predict(
                x[self.regressors_x_columns[i]])

            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i]
                               * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j] /
                               self.ensemble_n_estimators for j in range(len(curr_pred))]

        return predictions
