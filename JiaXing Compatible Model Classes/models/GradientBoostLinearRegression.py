import copy

from sklearn.model_selection import train_test_split

import numpy as np


from sklearn.linear_model import Lasso, LinearRegression



class GradientBoost_Lasso:

    def __init__(self, 
                 gb_n_estimators, 
                 gb_learning_rate, 
                 gb_max_features, 
                 gb_max_samples, 
                 alpha, 
                 random_state, 
                 **kwargs):

        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.gb_max_features = gb_max_features
        self.gb_max_samples = gb_max_samples

        self.alpha = alpha
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
            
            if self.alpha:
                lr = LinearRegression()
            else:
                lr = Lasso(alpha=self.alpha, 
                           random_state=self.random_state)
                

            lr.fit(sampled_train_x, sampled_train_y)

            self.regressors.append(lr)
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
            'gb_learning_rate': self.gb_learning_rate,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'gb_max_features': self.gb_max_features,
            'gb_max_samples': self.gb_max_samples
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self