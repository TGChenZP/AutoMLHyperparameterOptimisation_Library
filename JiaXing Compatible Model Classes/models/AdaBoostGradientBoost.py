from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from catboost import CatBoostRegressor


class ADABoost_ExplainableBoostingRegressor:

    def __init__(self,
                 ada_n_estimators,
                 ada_learning_rate,
                 max_bins,
                 min_sample_leaf,
                 interactions,
                 max_leaves,
                 learning_rate,
                 max_rounds,
                 n_jobs,
                 max_interaction_bins,
                 random_state,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins
        self.min_sample_leaf = min_sample_leaf
        self.interactions = interactions
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.n_jobs = n_jobs
        self.max_interaction_bins = max_interaction_bins

        self.random_state = random_state

    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator=ExplainableBoostingRegressor(max_bins=self.max_bins,
                                                                              min_sample_leaf=self.min_sample_leaf,
                                                                              interactions=self.interactions,
                                                                              max_leaves=self.max_leaves,
                                                                              learning_rate=self.learning_rate,
                                                                              max_rounds=self.max_rounds,
                                                                              max_interaction_bins=self.max_interaction_bins,
                                                                              n_jobs=self.n_jobs),
                                       n_estimators=self.ada_n_estimators,
                                       learning_rate=self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)

    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions


class ADABoost_CatBoostRegressor:

    def __init__(self,
                 ada_n_estimators,
                 ada_learning_rate,
                 n_estimators,
                 max_depth,
                 subsample,
                 colsample_bylevel,
                 max_bin,
                 reg_lambda,
                 learning_rate,
                 random_state,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.max_bin = max_bin
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator=CatBoostRegressor(n_estimators=self.n_estimators,
                                                                   max_depth=self.max_depth,
                                                                   subsample=self.subsample,
                                                                   colsample_bylevel=self.colsample_bylevel,
                                                                   max_bin=self.max_bin,
                                                                   reg_lambda=self.reg_lambda,
                                                                   learning_rate=self.learning_rate,
                                                                   random_state=self.random_state,
                                                                   verbose=False),
                                       n_estimators=self.ada_n_estimators,
                                       learning_rate=self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)

    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions


class ADABoost_LGBMRegressor:

    def __init__(self,
                 ada_n_estimators,
                 ada_learning_rate,
                 n_estimators,
                 max_depth,
                 subsample,
                 colsample_bytree,
                 learning_rate,
                 reg_alpha,
                 n_jobs,
                 random_state,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.learning_rate = learning_rate

        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator=LGBMRegressor(n_estimators=self.n_estimators,
                                                               max_depth=self.max_depth,
                                                               subsample=self.subsample,
                                                               colsample_bytree=self.colsample_bytree,
                                                               reg_alpha=self.reg_alpha,
                                                               learning_rate=self.learning_rate,
                                                               n_jobs=self.n_jobs,
                                                               random_state=self.random_state),
                                       n_estimators=self.ada_n_estimators,
                                       learning_rate=self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)

    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions


class ADABoost_XGBRegressor:
    def __init__(self,
                 ada_n_estimators,
                 ada_learning_rate,
                 n_estimators,
                 max_depth,
                 subsample,
                 colsample_bytree,
                 reg_alpha,
                 gamma,
                 eta,
                 n_jobs,
                 random_state,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.eta = eta
        self.reg_alpha = reg_alpha

        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator=XGBRegressor(n_estimators=self.n_estimators,
                                                              max_depth=self.max_depth,
                                                              subsample=self.subsample,
                                                              colsample_bytree=self.colsample_bytree,
                                                              gamma=self.gamma,
                                                              eta=self.eta,
                                                              reg_alpha=self.reg_alpha,
                                                              n_jobs=self.n_jobs,
                                                              random_state=self.random_state),
                                       n_estimators=self.ada_n_estimators,
                                       learning_rate=self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)

    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions


class ADABoost_GradientBoostingRegressor:

    def __init__(self,
                 ada_n_estimators,
                 ada_learning_rate,
                 n_estimators,
                 max_depth,
                 subsample,
                 max_features,
                 ccp_alpha,
                 learning_rate,
                 random_state,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator=GradientBoostingRegressor(n_estimators=self.n_estimators,
                                                                           max_depth=self.max_depth,
                                                                           subsample=self.subsample,
                                                                           max_features=self.max_features,
                                                                           ccp_alpha=self.ccp_alpha,
                                                                           learning_rate=self.learning_rate,
                                                                           random_state=self.random_state),
                                       n_estimators=self.ada_n_estimators,
                                       learning_rate=self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)

    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions


class ADABoost_HistGradientBoostingRegressor:

    def __init__(self,
                 ada_n_estimators,
                 ada_learning_rate,
                 max_depth,
                 max_bins,
                 interaction_cst,
                 learning_rate,
                 l2_regularization,
                 random_state,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.max_depth = max_depth
        self.max_bins = max_bins
        self.interaction_cst = interaction_cst
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator=HistGradientBoostingRegressor(max_depth=self.max_depth,
                                                                               max_bins=self.max_bins,
                                                                               interaction_cst=self.interaction_cst,
                                                                               learning_rate=self.learning_rate,
                                                                               l2_regularization=self.l2_regularization,
                                                                               random_state=self.random_state),
                                       n_estimators=self.ada_n_estimators,
                                       learning_rate=self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)

    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
