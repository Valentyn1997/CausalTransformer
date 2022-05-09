import torch
from torch import nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from copy import deepcopy
from typing import List


def grad_reverse(x, scale=1.0):

    class ReverseGrad(Function):
        """
        Gradient reversal layer
        """

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)


class FilteringMlFlowLogger(MLFlowLogger):
    def __init__(self, filter_submodels: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.filter_submodels = filter_submodels

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        params = deepcopy(params)
        [params.model.pop(filter_submodel) for filter_submodel in self.filter_submodels if filter_submodel in params.model]
        super().log_hyperparams(params)


def bce(treatment_pred, current_treatments, mode, weights=None):
    if mode == 'multiclass':
        return F.cross_entropy(treatment_pred.permute(0, 2, 1), current_treatments.permute(0, 2, 1), reduce=False, weight=weights)
    elif mode == 'multilabel':
        return F.binary_cross_entropy_with_logits(treatment_pred, current_treatments, reduce=False, weight=weights).mean(dim=-1)
    else:
        raise NotImplementedError()


class BRTreatmentOutcomeHead(nn.Module):
    """Used by CRN, EDCT, MultiInputTransformer"""

    def __init__(self, seq_hidden_units, br_size, fc_hidden_units, dim_treatments, dim_outcome, alpha=0.0, update_alpha=True,
                 balancing='grad_reverse'):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.balancing = balancing

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu1 = nn.ELU()

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.elu2 = nn.ELU()
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        self.linear4 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
        self.elu3 = nn.ELU()
        self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        self.treatment_head_params = ['linear2', 'linear3']

    def build_treatment(self, br, detached=False):
        if detached:
            br = br.detach()

        if self.balancing == 'grad_reverse':
            br = grad_reverse(br, self.alpha)

        br = self.elu2(self.linear2(br))
        treatment = self.linear3(br)  # Softmax is encapsulated into F.cross_entropy()
        return treatment

    def build_outcome(self, br, current_treatment):
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.elu3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, seq_output):
        br = self.elu1(self.linear1(seq_output))
        return br


class ROutcomeVitalsHead(nn.Module):
    """Used by G-Net"""

    def __init__(self, seq_hidden_units, r_size, fc_hidden_units, dim_outcome, dim_vitals, num_comp, comp_sizes):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.r_size = r_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_outcome = dim_outcome
        self.dim_vitals = dim_vitals
        self.num_comp = num_comp
        self.comp_sizes = comp_sizes

        self.linear1 = nn.Linear(self.seq_hidden_units, self.r_size)
        self.elu1 = nn.ELU()

        # Conditional distribution networks init
        self.cond_nets = []
        add_input_dim = 0
        for comp in range(self.num_comp):
            linear2 = nn.Linear(self.r_size + add_input_dim, self.fc_hidden_units)
            elu2 = nn.ELU()
            linear3 = nn.Linear(self.fc_hidden_units, self.comp_sizes[comp])
            self.cond_nets.append(nn.Sequential(linear2, elu2, linear3))

            add_input_dim += self.comp_sizes[comp]

        self.cond_nets = nn.ModuleList(self.cond_nets)

    def build_r(self, seq_output):
        r = self.elu1(self.linear1(seq_output))
        return r

    def build_outcome_vitals(self, r):
        vitals_outcome_pred = []
        for cond_net in self.cond_nets:
            out = cond_net(r)
            r = torch.cat((out, r), dim=-1)
            vitals_outcome_pred.append(out)
        return torch.cat(vitals_outcome_pred, dim=-1)


class AlphaRise(Callback):
    """
    Exponential alpha rise
    """
    def __init__(self, rate='exp'):
        self.rate = rate

    def on_epoch_end(self, trainer, pl_module) -> None:
        if pl_module.hparams.exp.update_alpha:
            assert hasattr(pl_module, 'br_treatment_outcome_head')
            p = float(pl_module.current_epoch + 1) / float(pl_module.hparams.exp.max_epochs)
            if self.rate == 'lin':
                pl_module.br_treatment_outcome_head.alpha = p * pl_module.br_treatment_outcome_head.alpha_max
            elif self.rate == 'exp':
                pl_module.br_treatment_outcome_head.alpha = \
                    (2. / (1. + np.exp(-10. * p)) - 1.0) * pl_module.br_treatment_outcome_head.alpha_max
            else:
                raise NotImplementedError()


def clip_normalize_stabilized_weights(stabilized_weights, active_entries, multiple_horizons=False):
    """
    Used by RMSNs
    """
    active_entries = active_entries.astype(bool)
    stabilized_weights[~np.squeeze(active_entries)] = np.nan
    sw_tilde = np.clip(stabilized_weights, np.nanquantile(stabilized_weights, 0.01), np.nanquantile(stabilized_weights, 0.99))
    if multiple_horizons:
        sw_tilde = sw_tilde / np.nanmean(sw_tilde, axis=0, keepdims=True)
    else:
        sw_tilde = sw_tilde / np.nanmean(sw_tilde)

    sw_tilde[~np.squeeze(active_entries)] = 0.0
    return sw_tilde
