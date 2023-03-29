import torch
from metrics.metric import Metric
from metrics.loss import Loss
from torch_scatter import scatter


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

def r2_score(y_true, y_pred):
    """
    Computes the R2 score between the predicted and true distributions
    Input:
        y_pred: tensor, predicted distribution
        y_true: tensor, true distribution
    Output:
        r2: float, R2 score between the predicted and true distributions
    """
    r2 = 1 - torch.sum((y_true - y_pred)**2) / torch.sum((y_true - torch.mean(y_true))**2)
    return r2

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
        name = name or 'accuracy'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)

        if self._name == 'accuracy':
            return torch.mean((y_pred == y_true).float())
        elif self._name == 'max_divergence':
            return max_divergence(y_pred, y_true)
        elif self._name == 'norm_divergence':
            return norm_divergence(y_pred, y_true)
        elif self._name == 'r2_score':
            return r2_score(y_true, y_pred)
        else:
            raise NotImplementedError
    

class MSE(Loss):
    def __init__(self, name=None):
        if name is None:
            name = 'mse'
        super().__init__(loss_fn=torch.nn.MSELoss(), name=name or 'mse')


class L1(Loss):
    def __init__(self, name=None):
        if name is None:
            name = 'l1'
        super().__init__(loss_fn=torch.nn.L1Loss(), name=name or 'l1')


class VorticityLoss(Loss):
    def __init__(self, name=None):
        if name is None:
            name = 'vorticity_loss'
        super().__init__(loss_fn=_compute_vorticity_loss, name=name or 'vorticity_loss')

    def _compute(self, y_pred, y_true, pos, edge_index, weight):
        return self.loss_fn(pos, y_pred, y_true, edge_index, weight)
    
    def compute(self, y_pred, y_true, pos, edge_index, weight):
        loss = self._compute(y_pred, y_true, pos, edge_index, weight)
        return loss


def _compute_vorticity_loss(pos, y_pred, y_true, edge_index, weight):
    """
    Computes the difference in vorticity between the predicted and true flow fields
    Input:
        y_pred: tensor, predicted distribution, in form [u, v, p]
        y_true: tensor, true distribution, in form [u, v, p]
    Output:
        vorticity_loss: float, vorticity loss between the predicted and true distributions
    """
    # Compute vorticity of predicted and true flow fields
    vorticity_pred = _compute_vorticity(y_pred, pos, edge_index)
    vorticity_true = _compute_vorticity(y_true, pos, edge_index)

    # Compute vorticity loss
    vorticity_loss = torch.nn.MSELoss()(vorticity_pred, vorticity_true)
    mse_loss = torch.nn.MSELoss()(y_pred, y_true)

    return weight * vorticity_loss + mse_loss

def _compute_grad(y, pos, edge_index):
    """
    Computes the vorticity of the flow field
    Input:
        y: tensor, flow field, in form [u, v, p]
        pos_x: tensor, x positions of the flow field nodes
        pos_y: tensor, y positions of the flow field nodes
    Output:
        vorticity: tensor, vorticity of the flow field
    """
    u, v = y[:, 0], y[:, 1]
    source_nodes = edge_index[0, :]
    target_nodes = edge_index[1, :]

    delta_x = pos[target_nodes, 0] - pos[source_nodes, 0]
    delta_y = pos[target_nodes, 1] - pos[source_nodes, 1]

    grad_u_x = scatter((u[target_nodes] - u[source_nodes]) / delta_x, source_nodes, dim=0, reduce='mean')
    grad_u_y = scatter((u[target_nodes] - u[source_nodes]) / delta_y, source_nodes, dim=0, reduce='mean')
    grad_v_x = scatter((v[target_nodes] - v[source_nodes]) / delta_x, source_nodes, dim=0, reduce='mean')
    grad_v_y = scatter((v[target_nodes] - v[source_nodes]) / delta_y, source_nodes, dim=0, reduce='mean')

    return grad_u_x, grad_u_y, grad_v_x, grad_v_y

def _compute_vorticity(y, pos, edge_index):
    """
    Computes the vorticity of the flow field
    Input:
        y: tensor, flow field, in form [u, v, p]
        pos_x: tensor, x positions of the flow field nodes
        pos_y: tensor, y positions of the flow field nodes
    Output:
        vorticity: tensor, vorticity of the flow field
    """
    grad_u_x, grad_u_y, grad_v_x, grad_v_y = _compute_grad(y, pos, edge_index)
    vorticity = grad_v_x - grad_u_y
    return vorticity


