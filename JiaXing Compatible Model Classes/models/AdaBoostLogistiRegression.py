from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

class ADABoost_LogisticRegression:

    def __init__(self, 
                 ada_n_estimators, 
                 ada_learning_rate, 
                 C,
                 l1_ratio,
                 random_state, 
                 solver = 'lbfgs',
                 max_iter = 1000,
                 multi_class = 'auto',
                 n_jobs = -1,
                 penalty= 'l1',
                 **kwargs):
        
        self.ada_n_estimators = ada_n_estimators
        self.ada_learning_rate = ada_learning_rate

        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.penalty = penalty



    def fit(self, train_x, train_y):

        
        self.model = AdaBoostClassifier(estimator = LogisticRegression(C=self.C,
                                                                       solver = self.solver,
                                                                       max_iter = self.max_iter,
                                                                       random_state = self.random_state,
                                                                       multi_class = self.multi_class,
                                                                       n_jobs = self.n_jobs,
                                                                       l1_ratio = self.l1_ratio,
                                                                       penalty= self.penalty), 
                                    n_estimators=self.ada_n_estimators, 
                                    learning_rate = self.ada_learning_rate,
                                    random_state=self.random_state)


        self.model.fit(train_x, train_y)


    def predict(self, x):

        predictions = self.model.predict(x)

        return predictions