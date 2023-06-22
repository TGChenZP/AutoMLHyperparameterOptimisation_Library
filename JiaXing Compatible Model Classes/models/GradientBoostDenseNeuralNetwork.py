import copy

from sklearn.model_selection import train_test_split

import numpy as np


from sklearn.neural_network import MLPRegressor



class GradientBoost_DNN_shrink_sk:

    def __init__(self, 
                 gb_n_estimators, 
                 gb_learning_rate, 
                 gb_max_features, 
                 gb_max_samples, 
                 n_hidden_layers, 
                 activation, 
                 alpha, 
                 batch_size, 
                 learning_rate, 
                 learning_rate_init, 
                 max_iter, 
                 random_state, 
                 **kwargs):

        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.gb_max_features = gb_max_features
        self.gb_max_samples = gb_max_samples

        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.regressors_x_columns = list()
        self.regressors = None

        

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(10000000, size = self.gb_n_estimators)

        self.regressors = []

        residual_y = copy.deepcopy(train_y)
        
        for i in range(self.gb_n_estimators):

            sampled_train_x, sampled_train_y, used_columns = self._sample_data_and_columns(train_x, residual_y, i)
            
            INPUT = len(sampled_train_x.columns)
            OUTPUT = 1
            gap = (INPUT-OUTPUT)//(self.n_hidden_layers+1)
            hidden_layer_sizes = [INPUT- i * gap for i in range(self.n_hidden_layers)]

            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                               activation=self.activation, 
                               alpha=self.alpha, 
                               batch_size=self.batch_size,
                               learning_rate=self.learning_rate, 
                               learning_rate_init=self.learning_rate_init, 
                               max_iter=self.max_iter, 
                               random_state=self.random_state)

            mlp.fit(sampled_train_x, sampled_train_y)

            self.regressors.append(mlp)
            self.regressors_x_columns.append(used_columns)

            curr_prediction = self.predict(train_x)

            if i != self.gb_n_estimators-1:
                residual_y = [train_y[j] - curr_prediction[j] for j in range(len(train_x))]

    

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(train_data, random_state = self.seeds[nth], train_size = self.gb_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        used_columns, unused_columns = train_test_split(list(sampled_train_x.columns), random_state = self.seeds[nth], train_size = self.gb_max_features)

        sampled_train_x = sampled_train_x[used_columns]

        return sampled_train_x, sampled_train_y, used_columns


    
    def predict(self, x):

        prediction = [0 for i in range(len(x))]
        for i in range(len(self.regressors)):

            tmp_prediction = self.regressors[i].predict(x[self.regressors_x_columns[i]])

            if i == 0:
                prediction = [prediction[j] + tmp_prediction[j] for j in range(len(x))]

            else:
                prediction = [prediction[j] + self.gb_learning_rate * tmp_prediction[j] for j in range(len(x))]
        
        return prediction
    

    def get_params(self, deep=False):
        return {
            'gb_n_estimators': self.gb_n_estimators,
            'learning_rate': self.learning_rate,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'gb_max_features': self.gb_max_features,
            'gb_max_samples': self.gb_max_samples
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self





class GradientBoost_DNN_const_sk:

    def __init__(self, gb_n_estimators, gb_learning_rate, gb_max_features, gb_max_samples, n_hidden_layers, activation, alpha, batch_size, learning_rate, learning_rate_init, max_iter, random_state, hidden_layer_n_neurons, **kwargs):
        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.gb_max_features = gb_max_features
        self.gb_max_samples = gb_max_samples
        self.hidden_layer_n_neurons = hidden_layer_n_neurons

        self.regressors_x_columns = list()
        self.regressors = None

        

    def fit(self, train_x, train_y):

        np.random.seed(self.random_state)
        self.seeds = np.random.randint(10000000, size = self.gb_n_estimators)

        self.regressors = []

        residual_y = copy.deepcopy(train_y)
        
        for i in range(self.gb_n_estimators):

            sampled_train_x, sampled_train_y, used_columns = self._sample_data_and_columns(train_x, residual_y, i)
            
            hidden_layer_sizes = [self.hidden_layer_n_neurons for i in range(self.n_hidden_layers)]

            dnn_c = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=self.activation, alpha=self.alpha, batch_size=self.batch_size,
                                  learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, random_state=self.random_state)
            
            dnn_c.fit(sampled_train_x, sampled_train_y)

            self.regressors.append(dnn_c)
            self.regressors_x_columns.append(used_columns)

            curr_prediction = self.predict(train_x)

            if i != self.gb_n_estimators-1:
                residual_y = [train_y[j] - curr_prediction[j] for j in range(len(train_x))]

    

    def _sample_data_and_columns(self, train_x, train_y, nth):

        train_data = copy.deepcopy(train_x)
        train_data['y'] = train_y

        sampled_train_data, sampled_dev_data = train_test_split(train_data, random_state = self.seeds[nth], train_size = self.gb_max_samples)

        sampled_train_x = sampled_train_data.drop(['y'], axis=1)
        sampled_train_y = sampled_train_data['y']

        used_columns, unused_columns = train_test_split(list(sampled_train_x.columns), random_state = self.seeds[nth], train_size = self.gb_max_features)

        sampled_train_x = sampled_train_x[used_columns]

        return sampled_train_x, sampled_train_y, used_columns


    
    def predict(self, x):

        prediction = [0 for i in range(len(x))]
        for i in range(len(self.regressors)):

            tmp_prediction = self.regressors[i].predict(x[self.regressors_x_columns[i]])

            if i == 0:
                prediction = [prediction[j] + tmp_prediction[j] for j in range(len(x))]

            else:
                prediction = [prediction[j] + self.gb_learning_rate * tmp_prediction[j] for j in range(len(x))]
        
        return prediction
    

    def get_params(self, deep=False):
        return {
            'gb_n_estimators': self.gb_n_estimators,
            'learning_rate': self.learning_rate,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'gb_max_features': self.gb_max_features,
            'gb_max_samples': self.gb_max_samples
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self