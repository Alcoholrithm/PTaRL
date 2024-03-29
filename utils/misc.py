from sklearn import metrics
import torch.nn.init as init
from abc import ABC, abstractmethod

class BaseScorer(ABC):
    def __init__(self, metric:str=None) -> None:
        """Base scorer for evaluation

        Args:
            metric (str): The name of the evaluation metric in sklearn.metrics
        """
        self.__name__ = metric
        self.metric = getattr(metrics, metric)
    
    @abstractmethod
    def __call__(self, y, y_hat) -> float:
        pass

class AccuracyScorer(BaseScorer):
    def __init__(self, metric:str = "accuracy_score") -> None:
        super().__init__(metric)
    
    def __call__(self, y, y_hat) -> float:
        return self.metric(y, y_hat.argmax(1))
    
class MSEScorer(BaseScorer):
    def __init__(self, metric: str="mean_squared_error") -> None:
        super().__init__(metric)
    
    def __call__(self, y, y_hat) -> float:
        return self.metric(y, y_hat)
    
def initialize_weights(model, initialization='kaiming_uniform', exclude_layers=["norm"]):
    """
    Initializes the weights and biases of the given PyTorch model using the specified initialization method.
    
    Args:
        model (nn.Module): PyTorch model instance.
        initialization (str): Name of the initialization method.
            Options: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'uniform', 'normal'.
            Defaults to 'xavier_uniform'.
    """

    for name, param in model.named_parameters():
        if any(layer_name.upper() in name.upper() for layer_name in exclude_layers):
            continue
        if 'weight' in name:
            if initialization == 'xavier_uniform':
                init.xavier_uniform_(param)
            elif initialization == 'xavier_normal':
                init.xavier_normal_(param)
            elif initialization == 'kaiming_uniform':
                init.kaiming_uniform_(param)
            elif initialization == 'kaiming_normal':
                init.kaiming_normal_(param)
            elif initialization == 'uniform':
                init.uniform_(param, -0.1, 0.1)  # Adjust range as needed
            elif initialization == 'normal':
                init.normal_(param, mean=0, std=0.01)  # Adjust mean and std as needed
            else:
                raise ValueError("Invalid initialization method. Please choose from 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'uniform', or 'normal'.")
        elif 'bias' in name:
            init.constant_(param, 0.0)  # Initialize biases to zero
