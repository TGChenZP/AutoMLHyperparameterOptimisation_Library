from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bays import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB





class ADABoost_GaussianNB:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 var_smoothing,
                 random_state,
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.var_smoothing = var_smoothing
        
        self.random_state = random_state
    


    def fit(self, train_x, train_y):

        self.model = AdaBoostClassifier(estimator = GaussianNB(var_smoothing = self.var_smoothing),
                                        random_state=self.random_state)
        

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    




class ADABoost_MultinomialNB:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha,
                 fit_prior,
                 random_state,
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.alpha = alpha
        self.fit_prior = fit_prior

        self.random_state = random_state
    


    def fit(self, train_x, train_y):

        self.model = AdaBoostClassifier(estimator = MultinomialNB(alpha = self.alpha,
                             fit_prior = self.fit_prior),
                             random_state=self.random_state)
        

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    




class ADABoost_ComplementNB:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha,
                 fit_prior,
                 random_state,
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.alpha = alpha
        self.fit_prior = fit_prior

        self.random_state = random_state
    


    def fit(self, train_x, train_y):

        self.model = AdaBoostClassifier(estimator = ComplementNB(alpha = self.alpha,
                             fit_prior = self.fit_prior),
                             random_state=self.random_state)
        

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    





class ADABoost_CategoricalNB:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha,
                 fit_prior,
                 random_state,
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.alpha = alpha
        self.fit_prior = fit_prior

        self.random_state = random_state
    


    def fit(self, train_x, train_y):

        self.model = AdaBoostClassifier(estimator = CategoricalNB(alpha = self.alpha,
                             fit_prior = self.fit_prior),
                             random_state=self.random_state)
        

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    




class ADABoost_BenoulliNB:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha,
                 fit_prior,
                 random_state,
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.alpha = alpha
        self.fit_prior = fit_prior

        self.random_state = random_state


    def fit(self, train_x, train_y):

        self.model = AdaBoostClassifier(estimator = BernoulliNB(alpha = self.alpha,
                             fit_prior = self.fit_prior),
                             random_state=self.random_state)
        

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions