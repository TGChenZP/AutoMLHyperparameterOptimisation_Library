from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import StackingRegressor





class StackerClassifier:

    def __init__(self, params, **kwargs):

        self.passthrough = params['passthrough']
        self.cv = params['cv']
        self.estimators = params['estimators']
        self.final_estimator = params['final_estimator']
        self.n_jobs = params.get('n_jobs', -1)
        self.stack_method = params.get('stack_method', 'auto')
        self.final_estimator_hp_combo = {hp:params[hp] for hp in params if hp not in ['passthrough', 'cv', 'estimators', 'final_estimator', 'n_jobs', 'stack_method']}

        print(self.final_estimator_hp_combo)



    def fit(self, train_x, train_y):

        self.model = StackingClassifier(estimators = self.estimators, 
                                        cv = self.cv, 
                                        passthrough=self.passthrough, 
                                        final_estimator=self.final_estimator(**self.final_estimator_hp_combo), 
                                        n_jobs=self.n_jobs, 
                                        stack_method=self.stack_method)

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    




class StackerRegressor:

    def __init__(self, params, **kwargs):

        self.passthrough = params['passthrough']
        self.cv = params['cv']
        self.estimators = params['estimators']
        self.final_estimator = params['final_estimator']
        self.n_jobs = params.get('n_jobs', -1)
        self.final_estimator_hp_combo = {hp:params[hp] for hp in params if hp not in ['passthrough', 'cv', 'estimators', 'final_estimator', 'n_jobs']}

        print(self.final_estimator_hp_combo)



    def fit(self, train_x, train_y):

        self.model = StackingRegressor(estimators = self.estimators, 
                                       cv = self.cv, 
                                       passthrough=self.passthrough, 
                                       final_estimator=self.final_estimator(**self.final_estimator_hp_combo), 
                                       n_jobs=self.n_jobs)

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    



## SAMPLE USAGE
# params = {'passthrough': True, 'n_jobs': -1, 'cv': 5, 'estimators':[('logistic_regression', LogisticRegression(C=1))], 'final_estimator':RandomForestClassifier, 'stack_method':'auto', 'n_estimators' : 10}
# model = StackerClassifier(params)