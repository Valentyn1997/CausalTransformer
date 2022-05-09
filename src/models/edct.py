from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, Subset
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns

from src.models.time_varying_model import BRCausalModel
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import BRTreatmentOutcomeHead
from src.models.utils_transformer import TransformerEncoderBlock, TransformerDecoderBlock, AbsolutePositionalEncoding, \
    RelativePositionalEncoding


logger = logging.getLogger(__name__)


class EDCT(BRCausalModel):

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'encoder', 'decoder'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection],
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.basic_block_cls = None
        self.save_hyperparameters(args)  # Will be logged to mlflow

    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of specific sub-network (Encoder/decoder)
        Args:
            sub_args: sub-network hyperparameters
        """
        try:
            self.max_seq_length = sub_args.max_seq_length
            self.br_size = sub_args.br_size  # balanced representation size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            # self.attn_dropout_rate = sub_args.attn_dropout_rate

            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.num_heads

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

            # Pytorch model init
            self.input_transformation = nn.Linear(self.input_size, self.seq_hidden_units) if self.input_size else None

            # Init of positional encodings
            self.self_positional_encoding = self.self_positional_encoding_k = self.self_positional_encoding_v = None
            if sub_args.self_positional_encoding.absolute:
                self.self_positional_encoding = \
                    AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                               sub_args.self_positional_encoding.trainable)
            else:
                # Relative positional encoding is shared across heads
                self.self_positional_encoding_k = \
                    RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.self_positional_encoding.trainable)
                self.self_positional_encoding_v = \
                    RelativePositionalEncoding(sub_args.self_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.self_positional_encoding.trainable)

            self.cross_positional_encoding = self.cross_positional_encoding_k = self.cross_positional_encoding_v = None
            if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
                self.cross_positional_encoding = \
                    AbsolutePositionalEncoding(self.max_seq_length, self.seq_hidden_units,
                                               sub_args.cross_positional_encoding.trainable)
            elif 'cross_positional_encoding' in sub_args:
                # Relative positional encoding is shared across heads
                self.cross_positional_encoding_k = \
                    RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.cross_positional_encoding.trainable, cross_attn=True)
                self.cross_positional_encoding_v = \
                    RelativePositionalEncoding(sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                                               sub_args.cross_positional_encoding.trainable, cross_attn=True)

            self.transformer_blocks = [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size,
                                                            self.seq_hidden_units * 4, self.dropout_rate, self.dropout_rate,
                                                            self_positional_encoding_k=self.self_positional_encoding_k,
                                                            self_positional_encoding_v=self.self_positional_encoding_v,
                                                            cross_positional_encoding_k=self.cross_positional_encoding_k,
                                                            cross_positional_encoding_v=self.cross_positional_encoding_v)
                                       for _ in range(self.num_layer)]
            self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
            self.output_dropout = nn.Dropout(self.dropout_rate)

            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(self.seq_hidden_units, self.br_size,
                                                                    self.fc_hidden_units, self.dim_treatments, self.dim_outcome,
                                                                    self.alpha, self.update_alpha, self.balancing)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.num_heads = new_args['num_heads']

        if 'seq_hidden_units' in new_args:  # Only relevant for encoder: seq_hidden_units should be divisible by num_heads
            # seq_hidden_units should even number - required for fixed positional encoding
            sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
            comon_multiplier = np.lcm.reduce([sub_args.num_heads, 2]).item()
            if sub_args.seq_hidden_units % comon_multiplier != 0:
                sub_args.seq_hidden_units = sub_args.seq_hidden_units + \
                    (comon_multiplier - sub_args.seq_hidden_units % comon_multiplier)
            print(f'Factual seq_hidden_units of {model_type}: {sub_args.seq_hidden_units}.')

        sub_args.br_size = int(input_size * new_args['br_size'])
        # br-size of encoder = seq_hidden_units of decoder should be divisible by num_heads
        if model_type == 'encoder' and model_args.train_decoder:
            if model_args.decoder.tune_hparams:
                # divisible by all possible num_heads in grid and even (for fixed positional encoding)
                comon_multiplier = np.lcm.reduce(model_args.decoder.hparams_grid.num_heads + [2]).item()
            else:
                comon_multiplier = np.lcm.reduce([model_args.decoder.num_heads, 2]).item()

            if sub_args.br_size % comon_multiplier != 0:
                sub_args.br_size = sub_args.br_size + (comon_multiplier - sub_args.br_size % comon_multiplier)
            print(f'Factual br_size of {model_type}: {sub_args.br_size }.')

        sub_args.fc_hidden_units = int(sub_args.br_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']

    def build_br(self, prev_treatments, vitals_or_prev_outputs, static_features, active_entries, encoder_br=None,
                 active_encoder_br=None):
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.input_transformation(x)

        if active_encoder_br is None and encoder_br is None:  # Only self-attention
            for block in self.transformer_blocks:

                if self.self_positional_encoding is not None:
                    x = x + self.self_positional_encoding(x)

                x = block(x, active_entries)

        else:  # Both self-attention and cross-attention
            assert x.shape[-1] == encoder_br.shape[-1]

            for block in self.transformer_blocks:

                if self.cross_positional_encoding is not None:
                    encoder_br = encoder_br + self.cross_positional_encoding(encoder_br)

                if self.self_positional_encoding is not None:
                    x = x + self.self_positional_encoding(x)

                x = block(x, encoder_br, active_entries, active_encoder_br)

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br

    def _visualize(self, fig_keys: list, dataset: Dataset, index=0, artifacts_path=None):
        figs_axes = {k: plt.subplots(ncols=self.num_heads, nrows=self.num_layer, squeeze=False,
                                     figsize=(6 * self.num_heads, 5 * self.num_layer)) for k in fig_keys}

        def plot_attn(attention, inp, out, layer, ax):
            p_attn = out[1]
            n_heads = p_attn.size(1)

            for j in range(n_heads):
                sns.heatmap(p_attn[0, j].cpu().numpy(), ax=ax[layer, j])
                ax[layer, j].title.set_text(f'Head {j} -- Layer {layer}')

        handles = []
        for i, transformer_block in enumerate(self.transformer_blocks):
            for k in fig_keys:
                att_layer = getattr(transformer_block, k).attention
                handles.append(att_layer.register_forward_hook(partial(plot_attn, layer=i, ax=figs_axes[k][1])))

        # Forward pass
        subset = Subset(dataset, [index])
        subset.subset_name = dataset.subset_name
        self.get_predictions(subset)

        for k in fig_keys:
            figs_axes[k][0].suptitle(f'{k}: {dataset.subset_name} datasets, datapoint index: {index}', fontsize=14)

        if artifacts_path is not None:
            for k in fig_keys:
                figs_axes[k][0].savefig(artifacts_path + f'/{self.model_type}_{k}_{index}.png')
        else:
            plt.show()

        for handle in handles:
            handle.remove()

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        """
        Vizualizes attention scores
        :param dataset: dataset
        :param index: index of an instance
        :param artifacts_path: Path for saving
        """
        fig_keys = ['self_attention']
        if self.model_type == 'decoder':
            fig_keys += ['cross_attention']

        self._visualize(fig_keys, dataset, index, artifacts_path)


class EDCTEncoder(EDCT):

    model_type = 'encoder'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.basic_block_cls = TransformerEncoderBlock
        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch['prev_treatments']
        vitals_or_prev_outputs = []
        vitals_or_prev_outputs.append(batch['vitals']) if self.has_vitals else None
        vitals_or_prev_outputs.append(batch['prev_outputs']) if self.autoregressive else None
        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        active_entries = batch['active_entries']

        br = self.build_br(prev_treatments, vitals_or_prev_outputs, static_features, active_entries)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br


class EDCTDecoder(EDCT):

    model_type = 'decoder'

    def __init__(self, args: DictConfig,
                 encoder: EDCTEncoder = None,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 encoder_r_size: int = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.basic_block_cls = TransformerDecoderBlock

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.encoder = encoder
        args.model.decoder.seq_hidden_units = self.encoder.br_size if encoder is not None else encoder_r_size
        self._init_specific(args.model.decoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder, save_encoder_r=True)
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch['prev_treatments']
        vitals_or_prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        encoder_br = batch['encoder_r']
        active_entries = batch['active_entries']
        active_encoder_br = batch['active_encoder_r']

        br = self.build_br(prev_treatments, vitals_or_prev_outputs, static_features, active_entries, encoder_br,
                           active_encoder_br)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br
