import numpy as np
import torch


class Metric:
    """
    Base class for all metrics.
    """
    def __init__(self, name):
        self._name = name

    def _compute(self, y_pred, y_true):
        """
        Fulfilled by subclasses.
        Args:
            y_pred: tensor, predicted distribution
            y_true: tensor, true distribution
        Returns:
            metric: float, metric value
        """
        raise NotImplementedError
    
    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric. Fulfilled by subclasses.
        Args:
            metrics (Tensor, numpy.ndarray or list): metric values
        Returns:
            worst_metric: float, worst metric value
        """
        raise NotImplementedError
    
    def best(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the best-case metric. Fulfilled by subclasses.
        Args:
            metrics (Tensor, numpy.ndarray or list): metric values
        Returns:
            best_metric: float, best metric value
        """
        raise NotImplementedError
    
    @property
    def name(self):
        return self._name
    
    @property
    def agg_metric_field(self):
        """
        The name of the key in the results dictionary returned by Metric.compute().
        This should correspond to the aggregate metric computed on all of y_pred and y_true,
        in contrast to a group-wise evaluation.
        """
        return f'{self.name}_all'
    
    def group_metric_field(self, group_idx):
        """
        The name of the keys corresponding to individual group evaluations
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'{self.name}_group:{group_idx}'
    
    @property
    def worst_group_metric_field(self):
        """
        The name of the keys corresponding to the worst-group metric
        in the results dictionary returned by Metric.compute_group_wise().
        """
        return f'{self.name}_wg'
    
    def compute(self, y_pred, y_true):
        """
        Computes the metric between y_pred and y_true.
        Args:
            y_pred: tensor, predicted distribution
            y_true: tensor, true distribution
        Returns:
            results: dict, dictionary of metric values
        """
        results = {}
        results[self.agg_metric_field] = self._compute(y_pred, y_true)
        return results
    
    def compute_group_wise(self, y_pred, y_true, group_ids):
        """
        Computes the metric between y_pred and y_true, grouped by group_ids.
        Args:
            y_pred: tensor, predicted distribution
            y_true: tensor, true distribution
            group_ids: tensor, group ids
        Returns:
            results: dict, dictionary of metric values
        """
        results = {}
        group_ids = group_ids.cpu().numpy()
        group_ids_unique = np.unique(group_ids)
        metrics = []
        for group_id in group_ids_unique:
            group_mask = group_ids == group_id
            group_y_pred = y_pred[group_mask]
            group_y_true = y_true[group_mask]
            group_metric = self._compute(group_y_pred, group_y_true)
            metrics.append(group_metric)
            results[self.group_metric_field(group_id)] = group_metric
        results[self.worst_group_metric_field] = self.worst(metrics)
        return results