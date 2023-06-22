import copy
from sklearn.model_selection import train_test_split
import numpy as np

from collections import defaultdict as dd

from sklearn.naive_bays import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB




class Ensemble_GaussianNB:

    def __init__(self, 
                 var_smoothing,
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 weighted = False, 
                 **kwargs):
        
        
        self.var_smoothing = var_smoothing

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
        self.weighted = weighted

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
            
            gnb = GaussianNB(var_smoothing = self.var_smoothing)

            gnb.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(gnb)
            self.regressors_train_score.append(gnb.score(sampled_train_x, sampled_train_y))
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

        predictions = list()

        tmp_prediction_list = []

        for i in range(self.ensemble_n_estimators):
            
            curr_pred = self.regressors[i].predict_proba(x[self.regressors_x_columns[i]])
            tmp_prediction_list.append(curr_pred)
        
        
        if self.weighted == True:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += self.regressors_weight[i]
                
            sorted_tally = self._sort_dict(tally)

            predictions.append(sorted_tally[0][0])

        else:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += 1
                
            sorted_tally = self._sort_dict(tally)
            # 未来：要用到baseline to deal with top

            predictions.append(sorted_tally[0][0])


        return predictions
    
    
    
    def _sort_dict(self, dictionary):
        
        items = list(dictionary.items())
        items.sort(key = lambda x:x[1])
        
        return items





class Ensemble_MultinomialNB:

    def __init__(self, 
                 alpha,
                 fit_prior,
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 weighted = False, 
                 **kwargs):
        
        
        self.alpha = alpha
        self.fit_prior = fit_prior

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
        self.weighted = weighted

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
            
            mnb = MultinomialNB(alpha = self.alpha,
                             fit_prior = self.fit_prior)

            mnb.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(mnb)
            self.regressors_train_score.append(mnb.score(sampled_train_x, sampled_train_y))
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

        predictions = list()

        tmp_prediction_list = []

        for i in range(self.ensemble_n_estimators):
            
            curr_pred = self.regressors[i].predict_proba(x[self.regressors_x_columns[i]])
            tmp_prediction_list.append(curr_pred)
        
        
        if self.weighted == True:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += self.regressors_weight[i]
                
            sorted_tally = self._sort_dict(tally)

            predictions.append(sorted_tally[0][0])

        else:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += 1
                
            sorted_tally = self._sort_dict(tally)
            # 未来：要用到baseline to deal with top

            predictions.append(sorted_tally[0][0])


        return predictions
    
    
    
    def _sort_dict(self, dictionary):
        
        items = list(dictionary.items())
        items.sort(key = lambda x:x[1])
        
        return items
    




class Ensemble_ComplementNB:

    def __init__(self, 
                 alpha,
                 fit_prior,
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 weighted = False, 
                 **kwargs):
        
        
        self.alpha = alpha
        self.fit_prior = fit_prior

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
        self.weighted = weighted

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
            
            cnb = ComplementNB(alpha = self.alpha,
                             fit_prior = self.fit_prior)

            cnb.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(cnb)
            self.regressors_train_score.append(cnb.score(sampled_train_x, sampled_train_y))
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

        predictions = list()

        tmp_prediction_list = []

        for i in range(self.ensemble_n_estimators):
            
            curr_pred = self.regressors[i].predict_proba(x[self.regressors_x_columns[i]])
            tmp_prediction_list.append(curr_pred)
        
        
        if self.weighted == True:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += self.regressors_weight[i]
                
            sorted_tally = self._sort_dict(tally)

            predictions.append(sorted_tally[0][0])

        else:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += 1
                
            sorted_tally = self._sort_dict(tally)
            # 未来：要用到baseline to deal with top

            predictions.append(sorted_tally[0][0])


        return predictions
    
    
    
    def _sort_dict(self, dictionary):
        
        items = list(dictionary.items())
        items.sort(key = lambda x:x[1])
        
        return items
    




class Ensemble_CategoricalNB:

    def __init__(self, 
                 alpha,
                 fit_prior,
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 weighted = False, 
                 **kwargs):
        
        
        self.alpha = alpha
        self.fit_prior = fit_prior

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
        self.weighted = weighted

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
            
            cnb = CategoricalNB(alpha = self.alpha,
                             fit_prior = self.fit_prior)

            cnb.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(cnb)
            self.regressors_train_score.append(cnb.score(sampled_train_x, sampled_train_y))
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

        predictions = list()

        tmp_prediction_list = []

        for i in range(self.ensemble_n_estimators):
            
            curr_pred = self.regressors[i].predict_proba(x[self.regressors_x_columns[i]])
            tmp_prediction_list.append(curr_pred)
        
        
        if self.weighted == True:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += self.regressors_weight[i]
                
            sorted_tally = self._sort_dict(tally)

            predictions.append(sorted_tally[0][0])

        else:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += 1
                
            sorted_tally = self._sort_dict(tally)
            # 未来：要用到baseline to deal with top

            predictions.append(sorted_tally[0][0])


        return predictions
    
    
    
    def _sort_dict(self, dictionary):
        
        items = list(dictionary.items())
        items.sort(key = lambda x:x[1])
        
        return items

    



class Ensemble_BenoulliNB:

    def __init__(self, 
                 alpha,
                 fit_prior,
                 ensemble_n_estimators = 100, 
                 ensemble_max_features = 1.0, 
                 ensemble_max_samples = 1.0, 
                 weighted = False, 
                 **kwargs):
        
        
        self.alpha = alpha
        self.fit_prior = fit_prior

        self.ensemble_n_estimators = ensemble_n_estimators
        self.ensemble_max_features = ensemble_max_features
        self.ensemble_max_samples = ensemble_max_samples
        self.weighted = weighted

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
            
            bnb = BernoulliNB(alpha = self.alpha,
                             fit_prior = self.fit_prior)

            bnb.fit(sampled_dev_x, sampled_dev_y)

            self.regressors.append(bnb)
            self.regressors_train_score.append(bnb.score(sampled_train_x, sampled_train_y))
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

        predictions = list()

        tmp_prediction_list = []

        for i in range(self.ensemble_n_estimators):
            
            curr_pred = self.regressors[i].predict_proba(x[self.regressors_x_columns[i]])
            tmp_prediction_list.append(curr_pred)
        
        
        if self.weighted == True:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += self.regressors_weight[i]
                
            sorted_tally = self._sort_dict(tally)

            predictions.append(sorted_tally[0][0])

        else:
            for j in range(len(curr_pred)):
                tally = dd(int)
                for i in range(tmp_prediction_list):
                    tally[tmp_prediction_list[i][j]] += 1
                
            sorted_tally = self._sort_dict(tally)
            # 未来：要用到baseline to deal with top

            predictions.append(sorted_tally[0][0])


        return predictions
    
    
    
    def _sort_dict(self, dictionary):
        
        items = list(dictionary.items())
        items.sort(key = lambda x:x[1])
        
        return items