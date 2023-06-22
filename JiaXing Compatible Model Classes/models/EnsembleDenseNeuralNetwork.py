import copy
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np




class Ensemble_DNN_shrink_sk:

    def __init__(self, 
                 alpha, 
                 random_state, 
                 n_hidden_layers, 
                 activation, 
                 batch_size,
                 learning_rate, 
                 learning_rate_init,
                 max_iter, 
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 ensemble_weighted = False, 
                 **kwargs):
        
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
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
            
            mlp.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(mlp)
            self.regressors_train_score.append(mlp.score(sampled_train_x, sampled_train_y))
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
            
            if self.self_weighted == True:
                predictions = [predictions[j] + self.regressors_weight[i] * curr_pred[j] for j in range(len(curr_pred))]
            else:
                predictions = [predictions[j] + curr_pred[j]/self.ensemble_n_estimators for j in range(len(curr_pred))]


        return predictions
    




class Ensemble_DNN_const_sk:

    def __init__(self, 
                 alpha, 
                 random_state,
                 n_hidden_layers, 
                 activation, 
                 batch_size,
                 learning_rate,
                 learning_rate_init, 
                 max_iter, 
                 hidden_layer_n_neurons, 
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 ensemble_weighted = False, 
                 **kwargs):
        
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.hidden_layer_n_neurons = hidden_layer_n_neurons

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_weighted = ensemble_weighted
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples

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
            
            hidden_layer_sizes = [self.hidden_layer_n_neurons for i in range(self.n_hidden_layers)]

            mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                               activation=self.activation, 
                               alpha=self.alpha, 
                               batch_size=self.batch_size,
                               learning_rate=self.learning_rate, 
                               learning_rate_init=self.learning_rate_init, 
                               max_iter=self.max_iter, 
                               random_state=self.random_state)
            
            mlp.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(mlp)
            self.regressors_train_score.append(mlp.score(sampled_train_x, sampled_train_y))
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