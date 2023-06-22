
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

class ADABoost_DecisionTree:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 max_depth, 
                 ccp_alpha, 
                 random_state, 
                 max_features,
                 **kwargs):

        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.max_features = max_features
    


    def fit(self, train_x, train_y):

        self.model = AdaBoostRegressor(estimator = DecisionTreeRegressor(max_depth = self.max_depth, 
                                                                         max_features = self.max_features, 
                                                                         ccp_alpha=self.ccp_alpha, 
                                                                         random_state=self.random_state), 
                                       n_estimators=self.ada_n_estimators, 
                                       learning_rate = self.ada_learning_rate,
                                       random_state=self.random_state)

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions