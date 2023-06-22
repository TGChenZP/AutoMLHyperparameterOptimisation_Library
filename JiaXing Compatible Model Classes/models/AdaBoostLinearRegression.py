
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso, LinearRegression

class ADABoost_Lasso:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha, 
                 random_state, 
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.alpha = alpha
        self.random_state = random_state
    


    def fit(self, train_x, train_y):

        if self.alpha == 0:
            self.model = AdaBoostRegressor(estimator = LinearRegression(), 
                                        n_estimators=self.ada_n_estimators, 
                                        learning_rate = self.ada_learning_rate)
        else:
            self.model = AdaBoostRegressor(estimator = Lasso(alpha = self.alpha, 
                                                             random_state=self.random_state), 
                                        n_estimators=self.ada_n_estimators, 
                                        learning_rate = self.ada_learning_rate,
                                        random_state=self.random_state)

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions