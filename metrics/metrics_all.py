import torch
from metric import Metric
from loss import Loss


def max_divergence(y_pred, y_true):
    """
    Computes the maximum divergence between the predicted and true distributions
    Input:
        y_pred: tensor, predicted distribution
        y_true: tensor, true distribution
    Output:
        max_div: float, maximum divergence between the predicted and true distributions
    """
    max_div = 1 - torch.max(torch.abs(y_pred - y_true)) / torch.max(y_true)
    return max_div

def norm_divergence(y_pred, y_true):
    """
    Computes the norm divergence between the predicted and true distributions
    Input:
        y_pred: tensor, predicted distribution
        y_true: tensor, true distribution
    Output:
        norm_div: float, norm divergence between the predicted and true distributions
    """
    norm_div = 1 - (torch.norm(y_pred) - torch.norm(y_true)) / torch.norm(y_true)
    return norm_div


class Accuracy(Metric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        self.name = name or 'accuracy'
        super().__init__(name=self.name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)

        if self.name == 'accuracy':
            return torch.mean((y_pred == y_true).float())
        elif self.name == 'max_divergence':
            return max_divergence(y_pred, y_true)
        elif self.name == 'norm_divergence':
            return norm_divergence(y_pred, y_true)
        else:
            raise NotImplementedError
    

class MSE(Loss):
    def __init__(self, name=None):
        if name is None:
            name = 'mse'
        super().__init__(loss_fn=torch.nn.MSELoss(), name=name or 'mse')