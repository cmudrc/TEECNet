import numpy as np
import torch
from metrics.metric import Metric


class Loss(Metric):
    """
    Base class for all losses.
    """
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        """
        Fulfilled by subclasses.
        Args:
            y_pred: tensor, predicted distribution
            y_true: tensor, true distribution
        Returns:
            loss: float, loss value
        """
        return self.loss_fn(y_pred, y_true)

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of losses, computes the worst-case loss. Fulfilled by subclasses.
        Args:
            losses (Tensor, numpy.ndarray or list): loss values
        Returns:
            worst_loss: float, worst loss value
        """
        return np.max(metrics)

    def best(self, metrics):
        """
        Given a list/numpy array/Tensor of losses, computes the best-case loss. Fulfilled by subclasses.
        Args:
            losses (Tensor, numpy.ndarray or list): loss values
        Returns:
            best_loss: float, best loss value
        """
        return np.min(metrics)

    @property
    def name(self):
        return self._name

    @property
    def agg_loss_field(self):
        """
        The name of the key in the results dictionary returned by Loss.compute().
        This should correspond to the aggregate loss computed on all of y_pred and y_true,
        in contrast to a group-wise evaluation.
        """
        return f'{self.name}_all'

    def group_loss_field(self, group_idx):
        """
        The name of the keys corresponding to individual group evaluations
        in the results dictionary returned by Loss.compute_group_wise().
        """
        return f'{self.name}_group_{group_idx}'

    def compute(self, y_pred, y_true):
        """
        Computes the loss on the entire y_pred and y_true.
        Args:
            y_pred: tensor, predicted distribution
            y_true: tensor, true distribution
        Returns:
            results: dict, dictionary of results
        """
        loss = self._compute(y_pred, y_true)
        # results = {self.agg_loss_field: loss}
        return loss

    def compute_group_wise(self, y_pred, y_true, group_idxs):
        """
        Computes the loss on each group in group_idxs.
        Args:
            y_pred: tensor, predicted distribution
            y_true: tensor, true distribution
            group_idxs: list, list of group indices
        Returns:
            results: dict, dictionary of results
        """
        results = {}
        for group_idx in group_idxs:
            group_loss = self._compute(y_pred, y_true, group_idx)
            results[self.group_loss_field(group_idx)] = group_loss
        return results