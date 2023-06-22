from sklearn.multioutput import MultiOutputRegressor


from sklearn.linear_model import Lasso

class MultiOutputLasso:

    def __init__(self, alpha = 1, random_state = None, **kwargs):

        self.clf = MultiOutputRegressor(Lasso(alpha=alpha, random_state=random_state))



    def fit(self, train_x, train_y):

        self.clf.fit(train_x, train_y)
    


    def predict(self, x):
        
        pred = self.clf.predict(x)

        return pred
    


from lightgbm import LGBMRegressor

class MultiOutputLightGBM:

    def __init__(self, n_estimators, max_depth, subsample, colsample_bytree, learning_rate, reg_alpha, **kwargs):

        self.clf = MultiOutputRegressor(LGBMRegressor(n_estimators = n_estimators, 
                                                      max_depth = max_depth, subsample = subsample, 
                                                      colsample_bytree = colsample_bytree, learning_rate = learning_rate, 
                                                      reg_alpha = reg_alpha))



    def fit(self, train_x, train_y):

        self.clf.fit(train_x, train_y)
    


    def predict(self, x):
        
        pred = self.clf.predict(x)

        return pred



from interpret.glassbox import ExplainableBoostingRegressor

class MultiOutputExplainableBoostingRegressor:

    def __init__(self, max_bins, min_samples_leaf, interactions, max_leaves, learning_rate, max_rounds, random_state, n_jobs, **kwargs):

        self.clf = MultiOutputRegressor(ExplainableBoostingRegressor(max_bins = max_bins,
                                                          min_samples_leaf = min_samples_leaf,
                                                          interactions = interactions,
                                                          max_leaves = max_leaves,
                                                          max_rounds = max_rounds,
                                                          learning_rate = learning_rate,
                                                          random_state = random_state,
                                                          n_jobs = n_jobs,
                                                          ))



    def fit(self, train_x, train_y):

        self.clf.fit(train_x, train_y)
    


    def predict(self, x):
        
        pred = self.clf.predict(x)

        return pred
    


from catboost import CatBoostRegressor

class MultiOutputCatBoostRegressor:

    def __init__(self, n_estimators, max_depth, subsample, colsample_bylevel, max_bin, reg_lambda, learning_rate, random_state, verbose, **kwargs):

        self.clf = MultiOutputRegressor(CatBoostRegressor(n_estimators = n_estimators,
                                                          max_depth = max_depth,
                                                          subsample = subsample,
                                                          colsample_bylevel = colsample_bylevel,
                                                          max_bin = max_bin,
                                                          reg_lambda = reg_lambda,
                                                          learning_rate = learning_rate,
                                                          random_state = random_state, verbose = verbose
                                                          ))



    def fit(self, train_x, train_y):

        self.clf.fit(train_x, train_y)
    


    def predict(self, x):
        
        pred = self.clf.predict(x)

        return pred
    
    


from sklearn.linear_model import PassiveAggressiveRegressor

class MultiOutputPassiveAggressiveRegressor:

    def __init__(self, C, epsilon, random_state = None, **kwargs):

        self.clf = MultiOutputRegressor(PassiveAggressiveRegressor(C=C, epsilon=epsilon, random_state=random_state))



    def fit(self, train_x, train_y):

        self.clf.fit(train_x, train_y)
    


    def predict(self, x):
        
        pred = self.clf.predict(x)

        return pred