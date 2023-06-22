import copy
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np




class Ensemble_Lasso:

    def __init__(self, 
                 alpha = 1.0, 
                 fit_intercept = True, 
                 precompute = False, 
                 copy_X = True, 
                 max_iter = 1000, 
                 tol = 1e-4, 
                 warm_start = False, 
                 positive = False, 
                 random_state = None, 
                 selection = 'cyclic', 
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 ensemble_weighted = False, 
                 n_jobs=-1, 
                 **kwargs):

        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
        self.ensemble_weighted = ensemble_weighted
        self.n_jobs = n_jobs

        self._initialise()



    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []



    def fit(self, train_x, train_y):
        
        np.random.seed(self.random_state)
        self.seeds = np.random.randint(10000000, size = self.ensemble_n_estimators)
        
        for i in range(self.ensemble_n_estimators):
            
            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(train_x, train_y, i)
            
            if self.alpha > 0:
                lr = Lasso(alpha = self.alpha, 
                           fit_intercept = self.fit_intercept, 
                           precompute = self.precompute, 
                           copy_X = self.copy_X, 
                           max_iter = self.max_iter, 
                           tol = self.tol, 
                           warm_start = self.warm_start, 
                           positive = self.positive, 
                           random_state = self.random_state, 
                           selection = self.selection)
            else:
                lr = LinearRegression(fit_intercept = self.fit_intercept, 
                                      copy_X = self.copy_X, 
                                      n_jobs = self.n_jobs, 
                                      positive = self.positive)

            lr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(lr)
            self.regressors_train_score.append(lr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [x/sum(self.regressors_train_score) for x in self.regressors_train_score]
            

    
    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(train_data, random_state = self.seeds[nth], train_size = self.ensemble_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(sampled_train_x.columns), random_state = self.seeds[nth], train_size = self.ensemble_max_features )

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

        

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.ensemble_n_estimators):
            
            curr_pred = self.regressors[i].predict(x[self.regressors_x_columns[i]])
            
            if self.ensemble_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i] * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j]/self.ensemble_n_estimators for j in range(len(curr_pred))]


        return predictions