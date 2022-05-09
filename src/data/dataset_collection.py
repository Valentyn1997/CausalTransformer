import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
import logging


logger = logging.getLogger(__name__)


class SyntheticDatasetCollection:
    """
    Dataset collection (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
    """

    def __init__(self, **kwargs):
        self.seed = None

        self.processed_data_encoder = False
        self.processed_data_decoder = False
        self.processed_data_multi = False
        self.processed_data_msm = False

        self.train_f = None
        self.val_f = None
        self.test_cf_one_step = None
        self.test_cf_treatment_seq = None
        self.train_scaling_params = None
        self.projection_horizon = None

        self.autoregressive = None
        self.has_vitals = None

    def process_data_encoder(self):
        self.train_f.process_data(self.train_scaling_params)
        self.val_f.process_data(self.train_scaling_params)
        self.test_cf_one_step.process_data(self.train_scaling_params)
        self.processed_data_encoder = True

    def process_propensity_train_f(self, propensity_treatment, propensity_history):
        """
        Generate stabilized weights for RMSN for the train subset
        Args:
            propensity_treatment: Propensity treatment network
            propensity_history: Propensity history network
        """
        prop_treat_train_f = propensity_treatment.get_propensity_scores(self.train_f)
        prop_hist_train_f = propensity_history.get_propensity_scores(self.train_f)
        self.train_f.data['stabilized_weights'] = np.prod(prop_treat_train_f / prop_hist_train_f, axis=2)

    def process_data_decoder(self, encoder, save_encoder_r=False):
        """
        Used by CRN, RMSN, EDCT
        """
        self.train_f.process_data(self.train_scaling_params)
        self.val_f.process_data(self.train_scaling_params)
        self.test_cf_treatment_seq.process_data(self.train_scaling_params)

        # Representation generation / One-step ahead prediction with encoder
        r_train_f = encoder.get_representations(self.train_f)
        r_val_f = encoder.get_representations(self.val_f)
        r_test_cf_treatment_seq = encoder.get_representations(self.test_cf_treatment_seq)
        outputs_test_cf_treatment_seq = encoder.get_predictions(self.test_cf_treatment_seq)

        # Splitting time series wrt specified projection horizon / Preparing test sequences
        self.train_f.process_sequential(r_train_f, self.projection_horizon, save_encoder_r=save_encoder_r)
        self.val_f.process_sequential(r_val_f, self.projection_horizon, save_encoder_r=save_encoder_r)
        # Considers only last timesteps according to projection horizon
        self.test_cf_treatment_seq.process_sequential_test(self.projection_horizon, r_test_cf_treatment_seq,
                                                           save_encoder_r=save_encoder_r)
        self.test_cf_treatment_seq.process_autoregressive_test(r_test_cf_treatment_seq, outputs_test_cf_treatment_seq,
                                                               self.projection_horizon, save_encoder_r=save_encoder_r)
        self.processed_data_decoder = True

    def process_data_multi(self):
        """
        Used by CT
        """
        self.train_f.process_data(self.train_scaling_params)
        if hasattr(self, 'val_f') and self.val_f is not None:
            self.val_f.process_data(self.train_scaling_params)
        self.test_cf_one_step.process_data(self.train_scaling_params)
        self.test_cf_treatment_seq.process_data(self.train_scaling_params)
        self.test_cf_treatment_seq.process_sequential_test(self.projection_horizon)
        self.test_cf_treatment_seq.process_sequential_multi(self.projection_horizon)

        self.processed_data_multi = True

    def split_train_f_holdout(self, holdout_ratio=0.1):
        """
        Used by G-Net
        """
        if not hasattr(self, 'train_f_holdout') and holdout_ratio > 0.0:
            self.train_f_holdout = deepcopy(self.train_f)
            for k, v in self.train_f.data.items():
                self.train_f.data[k], self.train_f_holdout.data[k] = train_test_split(v, test_size=holdout_ratio,
                                                                                      random_state=self.seed)
            logger.info(f'Splited train_f on train_f: {len(self.train_f)} and train_f_holdout: {len(self.train_f_holdout)}')

    def explode_cf_treatment_seq(self, mc_samples=1):
        """
        Producing mc_samples copies of test_cf_treatment_seq subset for further MC-Sampling (e.g. for G-Net)
        :param mc_samples: Number of copies
        """
        if not hasattr(self, 'test_cf_treatment_seq_mc'):
            logger.info(f'Exploding test_cf_treatment_seq {mc_samples} times')

            self.test_cf_treatment_seq_mc = []
            for m in range(mc_samples):
                self.test_cf_treatment_seq_mc.append(self.test_cf_treatment_seq)
                self.test_cf_treatment_seq_mc[m].data = deepcopy(self.test_cf_treatment_seq.data)


class RealDatasetCollection:
    """
    Dataset collection (train_f, val_f, test_f)
    """
    def __init__(self, **kwargs):
        self.seed = None

        self.processed_data_encoder = False
        self.processed_data_decoder = False
        self.processed_data_propensity = False
        self.processed_data_msm = False

        self.train_f = None
        self.val_f = None
        self.test_f = None
        self.train_scaling_params = None
        self.projection_horizon = None

        self.autoregressive = None
        self.has_vitals = None

    def process_data_encoder(self):
        pass

    def process_propensity_train_f(self, propensity_treatment, propensity_history):
        """
        Generate stabilized weights for RMSN for the train subset
        Args:
            propensity_treatment: Propensity treatment network
            propensity_history: Propensity history network
        """
        prop_treat_train_f = propensity_treatment.get_propensity_scores(self.train_f)
        prop_hist_train_f = propensity_history.get_propensity_scores(self.train_f)
        self.train_f.data['stabilized_weights'] = np.prod(prop_treat_train_f / prop_hist_train_f, axis=2)

    def process_data_decoder(self, encoder, save_encoder_r=False):
        """
        Used by CRN, RMSN, EDCT
        """
        # Multiplying test trajectories
        self.test_f.explode_trajectories(self.projection_horizon)

        # Representation generation / One-step ahead prediction with encoder
        r_train_f = encoder.get_representations(self.train_f)
        r_val_f = encoder.get_representations(self.val_f)
        r_test_f = encoder.get_representations(self.test_f)
        outputs_test_f = encoder.get_predictions(self.test_f)

        # Splitting time series wrt specified projection horizon / Preparing test sequences
        self.train_f.process_sequential(r_train_f, self.projection_horizon, save_encoder_r=save_encoder_r)
        self.val_f.process_sequential(r_val_f, self.projection_horizon, save_encoder_r=save_encoder_r)

        self.test_f.process_sequential_test(self.projection_horizon, r_test_f, save_encoder_r=save_encoder_r)
        self.test_f.process_autoregressive_test(r_test_f, outputs_test_f, self.projection_horizon, save_encoder_r=save_encoder_r)

        self.processed_data_decoder = True

    def process_data_multi(self):
        """
        Used by CT
        """
        self.test_f_multi = deepcopy(self.test_f)

        # Multiplying test trajectories
        self.test_f_multi.explode_trajectories(self.projection_horizon)

        self.test_f_multi.process_sequential_test(self.projection_horizon)
        self.test_f_multi.process_sequential_multi(self.projection_horizon)

        self.processed_data_multi = True

    def split_train_f_holdout(self, holdout_ratio=0.1):
        """
        Used by G-Net
        """
        if not hasattr(self, 'train_f_holdout') and holdout_ratio > 0.0:
            self.train_f_holdout = deepcopy(self.train_f)
            for k, v in self.train_f.data.items():
                self.train_f.data[k], self.train_f_holdout.data[k] = train_test_split(v, test_size=holdout_ratio,
                                                                                      random_state=self.seed)
            logger.info(f'Splited train_f on train_f: {len(self.train_f)} and train_f_holdout: {len(self.train_f_holdout)}')

    def explode_cf_treatment_seq(self, mc_samples=1):
        """
        Producing mc_samples copies of test_cf_treatment_seq subset for further MC-Sampling (e.g. for G-Net)
        :param mc_samples: Number of copies
        """
        if not hasattr(self, 'test_f_mc'):
            self.test_f_mc = []
            for m in range(mc_samples):
                logger.info(f'Exploding test_f {mc_samples} times')
                self.test_f_mc.append(self.test_f)
                self.test_f_mc[m].data = deepcopy(self.test_f.data)
