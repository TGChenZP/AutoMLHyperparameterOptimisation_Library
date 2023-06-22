import pandas as pd
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split





class BinomialRegression:

    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, **kwargs):
        
        self.PARAMETERS = ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state', 'solver', 'max_iter', 'multi_class', 'verbose', 'warm_start', 'n_jobs', 'l1_ratio']
        for key in kwargs:
            if key not in self.parameters:
                print("Failed to initiate")
                return
        
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

        self._initialise()



    def _initialise(self):

        self.regressors = []
        self.regressors_train_score = []
        self.regressors_weight = []
        self.regressors_x_columns = []
        self.model = None



    def fit(self, train_x, train_y):
        
    
            
        self.model = LogisticRegression(penalty=self.penalty, dual=self.dual, tol=self.tol,
                    C=self.C, fit_intercept=self.fit_intercept, intercept_scaling=self.intercept_scaling, class_weight=self.class_weight,
                    random_state=self.random_state, solver=self.solver, max_iter=self.max_iter, multi_class=self.multi_class, 
                    verbose=self.verbose, warm_start=self.warm_start, n_jobs=self.n_jobs, l1_ratio=self.l1_ratio)

        self.model.fit(train_x, train_y)

            
        

    def predict(self, x):

        tmp_predictions = self.model.predict_proba(x)
        predictions = [x[1] for x in tmp_predictions]

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
            'l1_ratio': self.l1_ratio
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self