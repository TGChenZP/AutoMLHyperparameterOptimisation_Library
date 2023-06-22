from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor





class ADABoost_DNN_shrink_sk:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha, 
                 random_state, 
                 n_hidden_layers, 
                 activation, 
                 batch_size,
                 learning_rate, 
                 learning_rate_init, 
                 max_iter,
                **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.random_state = random_state
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
    


    def fit(self, train_x, train_y):
        
        INPUT = len(train_x.columns)
        OUTPUT = 1
        gap = (INPUT-OUTPUT)//(self.n_hidden_layers+1)

        hidden_layer_sizes = [INPUT- i * gap for i in range(self.n_hidden_layers)]
        
        self.model = AdaBoostRegressor(MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                                    activation=self.activation, 
                                                    alpha=self.alpha, 
                                                    batch_size=self.batch_size,
                                                    learning_rate=self.learning_rate, 
                                                    learning_rate_init=self.learning_rate_init, 
                                                    max_iter=self.max_iter, 
                                                    random_state=self.random_state), 
                                        n_estimators=self.ada_n_estimators, 
                                        learning_rate = self.ada_learning_rate,
                                        random_state=self.random_state)

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions
    




class ADABoost_DNN_const_sk:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 alpha, 
                 random_state, 
                 n_hidden_layers, 
                 activation, 
                 batch_size,
                 learning_rate, 
                 learning_rate_init, 
                 max_iter, 
                 hidden_layer_n_neurons, 
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.hidden_layer_n_neurons = hidden_layer_n_neurons
    


    def fit(self, train_x, train_y):
        
        hidden_layer_sizes = [self.hidden_layer_n_neurons for i in range(self.n_hidden_layers)]
        
        self.model = AdaBoostRegressor(MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                                    activation=self.activation, 
                                                    alpha=self.alpha, 
                                                    batch_size=self.batch_size,
                                                    learning_rate=self.learning_rate, 
                                                    learning_rate_init=self.learning_rate_init, 
                                                    max_iter=self.max_iter, 
                                                    random_state=self.random_state), 
                                        n_estimators=self.ada_n_estimators, 
                                        learning_rate = self.ada_learning_rate,
                                        random_state=self.random_state)

        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions