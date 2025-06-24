from typing import Callable
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

def default_evaluation(model, X_val, y_val):
    """
    Default evaluation function that returns the accuracy of the model.
    This can be replaced with any custom evaluation function.
    """
    from sklearn.metrics import accuracy_score
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)
class Searcher:
    def __init__(self,train_data:pd.DataFrame, validation_data:pd.DataFrame, models, evaluation:Callable=default_evaluation,cv=5):
        """
        Initialize the Searcher with training and validation data, models, and evaluation function.

        :param train_data: Training data as a pandas DataFrame.
        :param validation_data: Validation data as a pandas DataFrame.
        :param models: List of models to evaluate.
        :param evaluation: Evaluation function to use for model performance assessment. Function signature should be
            foo(model, X_val, y_val) -> float, and it should handle the model evaluation logic from predict to score.
            The default is a function that returns the accuracy of the model.
        :type evaluation: Callable([model, X_val, y_val], float)
        :param cv: Number of cross-validation folds.
        """
        self.train_data = train_data
        self.validation_data = validation_data
        self.models = models
        self.cv = cv
        self.results = []
        self.evaluation = evaluation

    def search(self):
        pass


    def _evaluate_pipeline(self, model, evaluation):
        # Placeholder for pipeline evaluation logic
        return {
            "model": model,
            "evaluation": evaluation,
            "score": 0.0  # Replace with actual evaluation score
        }
class State:
    def __init__(self, state_dict):
        """
        Initialize a state with a dictionary of parameters.
        """
        self.state_dict = state_dict

    def __repr__(self):
        return f"State({self.state_dict})"

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.state_dict == other.state_dict

    def __hash__(self):
        return hash(tuple(sorted(self.state_dict.items())))
    
class StateNode:
    def __init__(self, state, train_data:pd.DataFrame, validation_data:pd.DataFrame, model, evaluation:Callable, cv=5,parent=None, classifier=False):
        """
        Initialize a state node in the search tree.
        """
        self.state:State = state
        self.parent = parent
        self.children = []
        self.is_terminal = False
        self.value = None
        self.train_data = train_data.copy()
        self.validation_data = validation_data.copy()
        self.model = model
        self.evaluation = evaluation

    def add_child(self, child_node):
        self.children.append(child_node)

    def set_terminal(self):
        self.is_terminal = True
    def evaluate(self):
        """
        Evaluate the current state node using the provided model and evaluation function.
        """
        if self.model and self.evaluation:
            skf = StratifiedKFold(n_splits=self.cv) if self.model.__class__.__name__ == 'Classifier' else None
            scores = []
            for train_index, val_index in skf.split(self.train_data, self.train_data['target']) if skf else [(range(len(self.train_data)), range(len(self.validation_data)))]:
                X_train, X_val = self.train_data.iloc[train_index], self.validation_data.iloc[val_index]
                y_train, y_val = X_train['target'], X_val['target']
                
                self.model.fit(X_train.drop('target', axis=1), y_train)
                score = self.evaluation(self.model, X_val.drop('target', axis=1), y_val)
                scores.append(score)
            self.value = np.mean(scores)
        else:
            raise ValueError("Model and evaluation function must be set to evaluate the state.")
    def __repr__(self):
        return f"StateNode(state={self.state}, is_terminal={self.is_terminal})"
