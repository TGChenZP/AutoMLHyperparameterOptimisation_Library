import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np




class Ensemble_BinomRegressor:

    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, n_estimators = 100, max_features = 1.0, max_samples = 1.0, weighted = False, **kwargs):
        
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.weighted = weighted

        self._initialise()



    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []



    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(10000000, size = self.n_estimators)
        
        for i in range(self.n_estimators):
            
            sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns = self._sample_data_and_columns(train_x, train_y, i)
            
            logr = LogisticRegression(penalty=self.penalty, dual=self.dual, tol=self.tol,
                        C=self.C, fit_intercept=self.fit_intercept, intercept_scaling=self.intercept_scaling, class_weight=self.class_weight,
                        random_state=self.random_state, solver=self.solver, max_iter=self.max_iter, multi_class=self.multi_class, 
                        verbose=self.verbose, warm_start=self.warm_start, n_jobs=self.n_jobs, l1_ratio=self.l1_ratio)

            logr.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(logr)
            self.regressors_train_score.append(logr.score(sampled_train_x, sampled_train_y))
            self.regressors_x_columns.append(used_columns)

        self.regressors_weight = [x/sum(self.regressors_train_score) for x in self.regressors_train_score]
            

    
    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(train_data, random_state = self.seeds[nth], train_size = self.max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        sampled_dev_x = sampled_dev_data.drop(['y'], axis=1)
        sampled_dev_y = sampled_dev_data['y']

        used_columns, unused_columns = train_test_split(list(sampled_train_x.columns), random_state = self.seeds[nth], train_size = self.max_features )

        sampled_train_x = sampled_train_x[used_columns]
        sampled_dev_x = sampled_dev_x[used_columns]

        return sampled_train_x, sampled_train_y, sampled_dev_x, sampled_dev_y, used_columns

        

    def predict(self, x):

        predictions = [0 for i in range(len(x))]

        for i in range(self.n_estimators):
            
            curr_pred = self.regressors[i].predict_proba(x[self.regressors_x_columns[i]])
            
            if self.weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i] * curr_pred[j][1] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j][1]/self.n_estimators for j in range(len(curr_pred))]


        return predictions
    

    def get_params(self, deep=False):
        return {
            'peanlty': self.penalty,
            'dual': self.dual,
            'tol': self.tol,
            'C': self.C,
            'fit_intercept': self.fit_intercept ,
            'intercept_scaling': self.intercept_scaling,
            'class_weight': self.class_weight ,
            'random_state': self.random_state ,
            'solver': self.solver ,
            'max_iter': self.max_iter,
            'multi_class': self.multi_class,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'l1_ratio': self.l1_ratio,
            'weighted': self.weighted,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'n_estimators': self.n_estimators
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self