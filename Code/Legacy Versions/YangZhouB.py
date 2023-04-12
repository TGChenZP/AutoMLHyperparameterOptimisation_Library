# 15/01/2023





import pandas as pd
import numpy as np
import statistics as s
import copy
import time
from itertools import combinations
import pickle
import random

from scipy.spatial.distance import cdist
from scipy.stats import t
from scipy import stats

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score





class YangZhouB:

    def __init__(self):
        """ Initialise class """
        self._initialise_objects()

        print('YangZhouB Initialised')



    def _initialise_objects(self):
        """ Helper to initialise objects """
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.tuning_result = None
        self.model = None
        self.parameter_choices = None
        self.hyperparameters = None
        self.checked = None
        self.result = None
        self.checked_core = None
        self.been_best = None
        self.been_cruised = None
        self.tuning_result_saving_address = None
        self.object_saving_address = None
        self._parameter_value_map_index = None
        self._seed = 19260817
        self.best_score = -np.inf
        self.best_combo = None
        self.best_clf = None
        self.clf_type = None
        self.n_items = None
        self._core = None
        self._cruise_indices = None
        self._cruise_indices_values = None
        self._cruise_combinations = None
        self._restarts = 0
        self._cruising = True
        self._surrounding_vectors = None

        self.regression_extra_output_columns = ['Train r2', 'Val r2', 'Test r2', 
            'Train RMSE', 'Val RMSE', 'Test RMSE', 'Train MAPE', 'Val MAPE', 'Test MAPE', 'Time']
        self.classification_extra_output_columns = ['Train accu', 'Val accu', 'Test accu', 
            'Train balanced_accu', 'Val balanced_accu', 'Test balanced_accu', 'Train f1', 'Val f1', 'Test f1', 
            'Train precision', 'Val precision', 'Test precision', 'Train recall', 'Val recall', 'Test recall', 'Time']



    def read_in_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        """ Reads in train validate test data for tuning """

        self.train_x = train_x
        print("Read in Train X data")

        self.train_y = train_y
        print("Read in Train x data")

        self.val_x = val_x
        print("Read in Val X data")

        self.val_y = val_y
        print("Read in Val y data")

        self.test_x = test_x
        print("Read in Test X data")

        self.test_y = test_y
        print("Read in Test y data")



    def read_in_model(self, model, type):
        """ Reads in underlying model object for tuning, and also read in what type of model it is """

        assert type == 'Classification' or type == 'Regression' # check

        # record
        self.model = model
        self.clf_type = type 

        print(f'Successfully read in model {self.model}, which is a {self.clf_type} model')



    def set_hyperparameters(self, parameter_choices):
        """ Input hyperparameter choices """

        self.parameter_choices = parameter_choices
        self._sort_hyperparameter_choices()

        self.hyperparameters = list(parameter_choices.keys())

        # automatically calculate how many different values in each hyperparameter
        self.n_items = [len(parameter_choices[key]) for key in self.hyperparameters]
        self.num_hyperparameters = {hyperparameter:len(parameter_choices[hyperparameter]) for hyperparameter in self.hyperparameters}

        # automatically setup checked and result arrays and tuning result dataframe
        self._get_checked_and_result_array()
        self._setup_tuning_result_df()

        print("Successfully recorded hyperparameter choices")


    
    def _sort_hyperparameter_choices(self):
        """ Helper to ensure all hyperparameter choice values are in order from lowest to highest """

        for key in self.parameter_choices:
            tmp = copy.deepcopy(list(self.parameter_choices[key]))
            tmp.sort()
            self.parameter_choices[key] = tuple(tmp)



    def _get_checked_and_result_array(self):
        """ Helper to set up checked and result array """

        self.checked = np.zeros(shape=self.n_items)
        self.result = np.zeros(shape=self.n_items)
        self.checked_core = np.zeros(shape=self.n_items)
        self.been_best = np.zeros(shape=self.n_items) # strictly for last part of Guidance Algorithm
        self.been_cruised = np.zeros(shape=self.n_items)



    def _setup_tuning_result_df(self):
        """ Helper to set up tuning result dataframe """

        tune_result_columns = copy.deepcopy(self.hyperparameters)

        # Different set of metric columns for different types of models
        if self.clf_type == 'Classification':
            tune_result_columns.extend(self.classification_extra_output_columns)
        elif self.clf_type == 'Regression':
            tune_result_columns.extend(self.regression_extra_output_columns)

        self.tuning_result = pd.DataFrame({col:list() for col in tune_result_columns})



    def _get_core(self):
        """ Helper to calculate core """
        self._core = [int(i/2) for i in self.n_items]



    def _get_cruise_combinations(self):
        """ Helper to cruise combinations """

        self._get_cruise_indices_values()
        self._generate_cruise_combinations() # first get cruise indicies, then use indicies to get combinations



    def _get_cruise_indices_values(self):
        """ Helper to get cruise indices values of each dimension which serves as building blocks for cruise combinations """

        self._cruise_indices = dict()
        for hyperparameter in self.hyperparameters:
            self._cruise_indices[hyperparameter] = self._get_cruise_indices_1d(d_val = self.num_hyperparameters[hyperparameter], max_jump = 5)

        self._cruise_indices_values = list(self._cruise_indices.values())



    def _get_cruise_indices_1d(self, d_val, max_jump = 5): 
        """ Helper that returns the appropriate cruise indices based on the number of values in dimension. Second argument controls maximum split size, defaulted to 5 """

        assert type(d_val) is int and type(max_jump) is int, "Error: type of input(s) is not int"
        assert max_jump >= 1, "Error: max_jump must be >= 1"

        gap = d_val - 1
        split = ((gap-1)//max_jump)

        jump = self._find_gaps(split, gap)

        cruise_indices_1d = self._find_cruise_indices_1d(jump)

        return cruise_indices_1d



    def _find_gaps(self, split, gap):
        """ Helper that finds the size of jumps between each element of the final cruise indices, as evenly split as possible with jump size <= 5 """

        if split > 0:
            jump = [gap//(split+1) for i in range(split+1)]
            diff = gap - sum(jump)
            if diff:
                for i in range(diff):
                    jump[i] += 1
        else:
            jump = [gap]

        return jump



    def _find_cruise_indices_1d(self, jump):
        """ Helper that finds the actual cruise_indices based on gaps """

        cruise_indices_1d = [0]
        for i in range(len(jump)):
            cruise_indices_1d.append(sum(jump[:i+1]))

        if cruise_indices_1d == [0, 0]:
            cruise_indices_1d = [0]
            
        return cruise_indices_1d



    def _generate_cruise_combinations(self):
        """ Helper that generates the actual cruise combinations based on cruise indicies """
        ##ALGORITHM: how to generate all combinations of any dimensions given each dimension has different values
        self._cruise_combinations = [[]]
        for i in range(len(self._cruise_indices_values)):

            tmp = copy.deepcopy(self._cruise_combinations)
            self._cruise_combinations = list()

            for x in tmp:

                for k in self._cruise_indices_values[i]:
                    y = copy.deepcopy(x)
                    
                    y.append(k)

                    self._cruise_combinations.append(y)



    def _sort_cruise_combos(self, max_combo):
        """ sort the cruise combos based on Euclidean distance from current max"""

        edist = list(cdist([max_combo], self._cruise_combinations).flatten())
        ordered_cruise_combos = [(self._cruise_combinations[i], edist[i]) for i in range(len(self._cruise_combinations))]

        ordered_cruise_combos.sort(reverse=True, key=lambda x: x[1])

        sorted_cruise_combos = [ordered_cruise_combos[i][0] for i in range(len(ordered_cruise_combos))]

        return sorted_cruise_combos

    

    def _get_max_surrounding_mean_sd(self):
        """ Helper to get the surrounding mean and sd given the current maximum """

        best_combo_surrounding_combos = self._get_surrounding_step_combos(self.best_combo)
        best_combo_surrounding_scores = [self.best_score]
        for combo in best_combo_surrounding_combos:
            best_combo_surrounding_scores.append(self.result[tuple(combo)])

        max_surrounding_mean = s.mean(best_combo_surrounding_scores)
        max_surrounding_sd = s.stdev(best_combo_surrounding_scores)

        return max_surrounding_mean, max_surrounding_sd



    def _cruise_warning_threshold(self, max_surrounding_mean, max_surrounding_sd, max_surrounding_n):
        """ Helper that gets the warning threshold by (mean of best_combo surrounds - halfwidth) """

        # use 0.95 (one sided test)
        qt = t.ppf(0.95, max_surrounding_n-1) 
        halfwidth = max_surrounding_sd * qt * 1/np.sqrt(max_surrounding_n)

        return max_surrounding_mean - halfwidth



    def _CruiseSystem(self):
        """ Helper that performs cruising """

        print(f"Cruising: round {self._restarts}\n") 

        # get cruise combos in sorted order (furthest away from current max)
        sorted_cruise_combos = self._sort_cruise_combos(self.best_combo)

        # calculate warning threshold by getting max_surrounding_sd first
        max_surrounding_mean, max_surrounding_sd = self._get_max_surrounding_mean_sd()

        warning_threshold = self._cruise_warning_threshold(max_surrounding_mean, max_surrounding_sd, len(self._surrounding_vectors))

        # check each cruise combo
        for cruise_combo in sorted_cruise_combos:

            # only search if it hasn't been cruised before (if has then is not an artifect of significance)
            if not self.been_cruised[tuple(cruise_combo)]:
                
                self.been_cruised[tuple(cruise_combo)] = 2 # actually been cruised

                # if above warning threshold, then stop cruise and restart guide
                if self.result[tuple(cruise_combo)] >= warning_threshold:
                   
                    print(f"Cruise suspended due to suspicious case")
                    
                    return cruise_combo


        # if reach here then all cruise indicies checked. can safely say end cruise
        self._cruising = False

        return



    def _get_core(self):
        """ Helper to calculate core """

        self._core = [int(i/2) for i in self.n_items]



    def _new_combos(self, core, vector):
        """ Helper that gets particular COORDINATE using a move in direction of VECTOR from particular CORE """

        assert len(core) == len(vector)

        new_combo = list()
        for i in range(len(vector)):
            val = core[i] + vector[i]
            if val >= self.n_items[i] or val < 0: # check whether combo is still in the field
                return False
            new_combo.append(val)

        return new_combo



    def _xlnx(self, x):
        """ Helper that returns x*ln(x), rounding up to next int """
        y = int(x*np.log(x))+1
        
        return y



    def _find_new_core(self, surrounding_combos, core):
        """ Helper that finds new cores - only those candidates with difference between core and treatment < 0.005 """

        new_cores = []
        tmp_new_cores = list()
        
        for i in range(len(surrounding_combos)): 

            diff = self.result[tuple(core)] - self.result[tuple(surrounding_combos[i])]

            if diff <= 0.005:
                if self.checked_core[ tuple(surrounding_combos[i]) ] == 0:
                    tmp_new_cores.append([tuple(surrounding_combos[i]), diff])
                        
        # # sort the tmp new cores list according to p values
        # tmp_new_cores.sort(key = lambda x:x[1])

        # # calculate how many new cores to accept according to the dimension of the grid (x = dim; accept x*ln(x))
        # n_accept = self._xlnx(len(core))
        
        for i in range(len(tmp_new_cores)):
            # if i >= n_accept:
            #     break

            new_cores.append(tmp_new_cores[i][0])
            self.checked_core[ tmp_new_cores[i][0] ] = 1

        return new_cores



    def _get_surrounding_step_vectors(self, core):
        """ find all horizontal steps """

        all_steps = list()

        for i in range(len(core)):
            for val in [-1, 1]:
                tmp = [0 for i in range(len(core))]
                tmp[i] = val
        
                all_steps.append(tmp)
        
        return all_steps
    


    def _get_surrounding_step_combos(self, core):
        """ find all vectors that are one horizontal steps away from core """

        surrounding_combos = list()
        for step in self._surrounding_vectors:

            add = 1
            candidate_surrounding_combo = list()

            for i in range(len(core)):
                new_index = core[i]+step[i]
                if new_index < 0 or new_index >= self.n_items[i]:
                    add = 0
                    break
                else:
                    candidate_surrounding_combo.append(new_index)

            if add:    
                surrounding_combos.append(candidate_surrounding_combo)

        return surrounding_combos



    def _get_new_cores(self, core):
        """ Helper that gets new cores given old cores """

        # if (should be rare) case where core has been a core before, then skip. For prevention of infinite loops
        # 2 means actual checked core, 1 means appended to checked core list but not checked
        if self.checked_core[tuple(core)] == 2:
            return
        else:
            self.checked_core[tuple(core)] = 2 # actually been checked as a core

        surrounding_combos = self._get_surrounding_step_combos(core)

        # actually tune the surrounding combos
        for combo in surrounding_combos:
            
            if self.checked[tuple(combo)] == 0:
                self._train_and_test_combo(combo)
            else:
                print(f'\tAlready Trained and Tested combination {combo}')

        # perform welch test and return surrounding combos that should be used as new core
        new_cores = self._find_new_core(surrounding_combos, core)

        return new_cores  



    def _GuidanceSystem(self, core):
        """ Helper that performs guidance search """

        if self._restarts == 0:
            print("Guidance: initial round \n")
        else:
            print("Guidance: round", self._restarts, '\n')

        print('\tround', self._restarts, 'iteration: ', 0, '\n')

        # first get a surrounding 3^d tuned
        new_cores = self._get_new_cores(core)
        self.been_cruised[tuple(core)] = 1 # represent don't need to be added as a to cruised - but not cruised, rather a core

        round = 1
        while new_cores: # while new cores are being added
            print('\tround', self._restarts, "iteration: ", round, "\n") 
            round += 1

            old_new_cores = copy.deepcopy(new_cores)
            new_cores = list()

            # for each of the new cores, 'recursively' tune and grab new cores; but each Iteration doesn't end until all cores of current round has been checked
            for new_core in old_new_cores:
                
                new_new_cores = self._get_new_cores(new_core)

                if not new_new_cores:
                    new_new_cores = []

                self.been_cruised[tuple(core)] == 1
                
                for new_new_core in new_new_cores:
                    if self.checked_core[tuple(new_new_core)] == 0:
                        new_cores.append(new_new_core)
                        self.checked_core[tuple(new_new_core)] = 1 # represent added to checked core - to prevent repeated added to core


        # for current max, get 3^d block. if new max happens to be found, continue to do 3^d block until no new max is found
        # just a cheap way to flesh out the max (the goal of YangZhou)
        while self.been_best[tuple(self.best_combo)] == 0:

            self.been_best[tuple(self.best_combo)] = 1 
    
            surrounding_combos = self._get_surrounding_step_combos(self.best_combo)
            for combo in surrounding_combos:
                
                if self.checked[tuple(combo)] == 0:
                    self._train_and_test_combo(combo)
                else:
                    print(f'\tAlready Trained and Tested combination {combo}')

        # print information of this round 

        print('% Combos Checked Thus Far:', int(sum(self.checked.reshape((np.prod(self.n_items))))), 'out of', np.prod(self.n_items), 'which is', f'{np.mean(self.checked).round(8)*100}%')



    def tune(self):
        """ Begin tuning """

        if self.train_x is None or self.train_y is None or self.val_x is None or self.val_y is None or self.test_x is None or self.test_y is None:
            print(" Missing one of the datasets, please run .read_in_data() ")
            return

        if self.model is None:
            print(" Missing model, please run .read_in_model() ")
            return

        if self.tuning_result_saving_address is None:
            print("Missing tuning result csv saving address, please run ._save_tuning_result() first")


        print("BEGIN TUNING\n\n") 
        
        # FIRST: get all cruise combinations as well as core, and tune all these
        self._get_core()
        self._get_cruise_combinations() 

        first_round_combinations = copy.deepcopy(self._cruise_combinations)
        first_round_combinations.append(self._core) 

        random.seed(self._seed)
        random.shuffle(first_round_combinations)

        print("STAGE ZERO: Tune all Cruise combinations\n\n")
        for combo in first_round_combinations:
            
            if not self.checked[tuple(combo)]:
                
                self._train_and_test_combo(combo)
            
            else:
                print(f'\tAlready Trained and Tested combination {combo}')
        

        # SECOND: from the core combo, begin guidance system
        self._surrounding_vectors = self._get_surrounding_step_vectors(self._core)

        print('\n')
        print("STAGE ONE: Begin initial Guidance system\n\n")

        self._restarts = 0
        self._GuidanceSystem(self._core) # Initial Round of Guidance
        self._restarts += 1

        # THIRD: Recursively Cruise and restart Guide if find a combo that is within halfwidth of mean of best combo surrounds
        print("STAGE TWO: Begin Cruise system\n\n")
        self._cruising = True
        while self._cruising:
            suspicious_case_combo = self._CruiseSystem()

            if self._cruising:
                self._GuidanceSystem(tuple(suspicious_case_combo))
                self._restarts += 1

        # FINALLY: Final extensive guidance search around maxes.
        print("FINAL STAGE: Begin final Guidance system\n\n")
        old_best_score = copy.deepcopy(self.best_score)
        self._restarts = 'FINAL'

        self._GuidanceSystem(self.best_combo)

        while(self.best_score-old_best_score > 0):
            old_best_score = copy.deepcopy(self.best_score)
            self._GuidanceSystem(self.best_combo)



        # Display final information
        print("TUNING FINISHED\n")

        print('Max Score: \n', self.best_score)
        print('Max Combo: \n', self.best_combo)

        print('% Combos Checked:', int(sum(self.checked.reshape((np.prod(self.n_items))))), 'out of', np.prod(self.n_items), 'which is', f'{np.mean(self.checked).round(8)*100}%')
    


    def _train_and_test_combo(self, combo):
        """ Helper to train and test each combination as part of tune() """

        combo = tuple(combo)
        
        params = {self.hyperparameters[i]:self.parameter_choices[self.hyperparameters[i]][combo[i]] for i in range(len(self.hyperparameters))}

        # initialise object
        clf = self.model(**params)

        # get time and fit
        start = time.time()
        clf.fit(self.train_x, self.train_y)
        end = time.time()

        # get predicted labels/values for three datasets
        train_pred = clf.predict(self.train_x)
        val_pred = clf.predict(self.val_x)
        test_pred = clf.predict(self.test_x)

        # get scores and time used
        time_used = end-start

        # build output dictionary and save result
        df_building_dict = params

        if self.clf_type == 'Regression':
            train_score = r2_score(self.train_y, train_pred)
            val_score = r2_score(self.val_y, val_pred)
            test_score = r2_score(self.test_y, test_pred)

            train_rmse = np.sqrt(mean_squared_error(self.train_y, train_pred))
            val_rmse = np.sqrt(mean_squared_error(self.val_y, val_pred))
            test_rmse = np.sqrt(mean_squared_error(self.test_y, test_pred))

            train_mape = mean_absolute_percentage_error(self.train_y, train_pred)
            val_mape = mean_absolute_percentage_error(self.val_y, val_pred)
            test_mape = mean_absolute_percentage_error(self.test_y, test_pred)

            df_building_dict['Train r2'] = [np.round(train_score, 4)]
            df_building_dict['Val r2'] = [np.round(val_score, 4)]
            df_building_dict['Test r2'] = [np.round(test_score, 4)]
            df_building_dict['Train RMSE'] = [np.round(train_rmse, 4)]
            df_building_dict['Val RMSE'] = [np.round(val_rmse, 4)]
            df_building_dict['Test RMSE'] = [np.round(test_rmse, 4)]
            df_building_dict['Train MAPE'] = [np.round(train_mape, 4)]
            df_building_dict['Val MAPE'] = [np.round(val_mape, 4)]
            df_building_dict['Test MAPE'] = [np.round(test_mape, 4)]
            df_building_dict['Time'] = [np.round(time_used, 2)]

        
        elif self.clf_type == 'Classification':
            train_score = accuracy_score(self.train_y, train_pred)
            val_score = accuracy_score(self.val_y, val_pred)
            test_score = accuracy_score(self.test_y, test_pred)

            train_bal_accu = balanced_accuracy_score(self.train_y, train_pred)
            val_bal_accu = balanced_accuracy_score(self.val_y, val_pred)
            test_bal_accu = balanced_accuracy_score(self.test_y, test_pred)

            train_f1 = f1_score(self.train_y, train_pred, average='weighted')
            val_f1 = f1_score(self.val_y, val_pred, average='weighted')
            test_f1 = f1_score(self.test_y, test_pred, average='weighted')

            train_precision = precision_score(self.train_y, train_pred, average='weighted')
            val_precision = precision_score(self.val_y, val_pred, average='weighted')
            test_precision = precision_score(self.test_y, test_pred, average='weighted')
        
            train_recall = recall_score(self.train_y, train_pred, average='weighted')
            val_recall = recall_score(self.val_y, val_pred, average='weighted')
            test_recall = recall_score(self.test_y, test_pred, average='weighted')

            df_building_dict['Train accu'] = [np.round(train_score, 4)]
            df_building_dict['Val accu'] = [np.round(val_score, 4)]
            df_building_dict['Test accu'] = [np.round(test_score, 4)]
            df_building_dict['Train balanced_accuracy'] = [np.round(train_bal_accu, 4)]
            df_building_dict['Val balanced_accuracy'] = [np.round(val_bal_accu, 4)]
            df_building_dict['Test balanced_accuracy'] = [np.round(test_bal_accu, 4)]
            df_building_dict['Train f1'] = [np.round(train_f1, 4)]
            df_building_dict['Val f1'] = [np.round(val_f1, 4)]
            df_building_dict['Test f1'] = [np.round(test_f1, 4)]
            df_building_dict['Train precision'] = [np.round(train_precision, 4)]
            df_building_dict['Val precision'] = [np.round(val_precision, 4)]
            df_building_dict['Test precision'] = [np.round(test_precision, 4)]
            df_building_dict['Train recall'] = [np.round(train_recall, 4)]
            df_building_dict['Val recall'] = [np.round(val_recall, 4)]
            df_building_dict['Test recall'] = [np.round(test_recall, 4)]
            df_building_dict['Time'] = [np.round(time_used, 2)]

        tmp = pd.DataFrame(df_building_dict)

        self.tuning_result = self.tuning_result.append(tmp)
        self._save_tuning_result()

        # update best score stats
        if val_score > self.best_score: 
            self.best_score = val_score
            self.best_clf = clf
            self.best_combo = combo

        # update internal governing DataFrames
        self.checked[combo] = 1
        self.result[combo] = val_score

        print(f'''\tTrained and Tested a Combo, taking {np.round(time_used, 2)} seconds
            Current best combo: {self.best_combo} with val score {self.best_score}''')


                
    def _save_tuning_result(self):
        """ Helper to export tuning result csv """

        tuning_result_saving_address_split = self.tuning_result_saving_address.split('.csv')[0]

        self.tuning_result.to_csv(f'{tuning_result_saving_address_split}.csv', index=False)

    

    def view_best_combo_and_score(self):
        """ View best combination and its validation score """
        
        print(f'(Current) Best combo: {self.best_combo} with val score {self.best_score}')



    def read_in_tuning_result_df(self, address): 
        """ Read in tuning result csv and read data into checked and result arrays """

        if self.parameter_choices is None:
            print("Missing parameter_choices to build parameter_value_map_index, please run set_hyperparameters() first")

        if self.clf_type is None:
            print('Missing clf_type. Please run .read_in_model() first.')

        self.tuning_result = pd.read_csv(address)
        print(f"Successfully read in tuning result of {len(self.tuning_result)} rows")

        self._create_parameter_value_map_index()

        # read DataFrame data into internal governing DataFrames of YangZhou
        for row in self.tuning_result.iterrows():
    
            combo = tuple([self._parameter_value_map_index[hyperparam][row[1][hyperparam]] for hyperparam in self.hyperparameters])
            
            self.checked[combo] = 1
            
            if self.clf_type == 'Regression':
                self.result[combo] = row[1]['Val r2']
            elif self.clf_type == 'Classification':
                self.result[combo] = row[1]['Val accu']
        
            # update best score stats
            if self.result[combo] > self.best_score: 
                self.best_score = self.result[combo]
                self.best_clf = None
                print(f"As new Best Combo {combo} is read in, best_clf is set to None")
                self.best_combo = combo


    
    def _create_parameter_value_map_index(self):
        """ Helper to create parameter-value index map """

        self._parameter_value_map_index = dict()
        for key in self.parameter_choices.keys():
            tmp = dict()
            for i in range(len(self.parameter_choices[key])):
                tmp[self.parameter_choices[key][i]] = i
            self._parameter_value_map_index[key] = tmp


    
    def set_tuning_result_saving_address(self, address):
        """ Read in where to save tuning object """

        self.tuning_result_saving_address = address
        print('Successfully set tuning output address')



    def _set_object_saving_address(self, address):
        """ Read in where to save the YangZhou object """

        self.object_saving_address = address
        print('Successfully set object output address')



    def export_yangzhou(self, address):
        """ Export yangzhou object """

        self._set_object_saving_address(address)

        # copy object and set big objects to None
        object_save = copy.deepcopy(self)
        
        object_save.train_x = None
        object_save.train_y = None
        object_save.val_x = None
        object_save.val_y = None
        object_save.test_x = None
        object_save.test_y = None

        # Export
        object_saving_address_split = self.object_saving_address.split('.pickle')[0]

        with open(f'{object_saving_address_split}.pickle', 'wb') as f:
            pickle.dump(object_save, f)

        print(f'Successfully exported YangZhou object as {self.object_saving_address}')