import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import AlphaRise, FilteringMlFlowLogger
from src.models.rmsn import RMSN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for RMSNs
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data to calculate dim_outcomes, dim_treatments, dim_vitals and dim_static_features
    seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    assert args.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
    dataset_collection.process_data_encoder()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    prop_treatment_callbacks, propensity_history_callbacks, encoder_callbacks, decoder_callbacks = [], [], [], []

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=RMSN.possible_model_types, experiment_name=experiment_name,
                                           tracking_uri=args.exp.mlflow_uri)
        encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        prop_treatment_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        propensity_history_callbacks += [LearningRateMonitor(logging_interval='epoch')]
    else:
        mlf_logger = None

    # ============================== Nominator (treatment propensity network) ==============================
    propensity_treatment = instantiate(args.model.propensity_treatment, args, dataset_collection, _recursive_=False)
    if args.model.propensity_treatment.tune_hparams:
        propensity_treatment.finetune(resources_per_trial=args.model.propensity_treatment.resources_per_trial)

    propensity_treatment_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                           logger=mlf_logger,
                                           max_epochs=args.exp.max_epochs,
                                           callbacks=prop_treatment_callbacks,
                                           gradient_clip_val=args.model.propensity_treatment.max_grad_norm,
                                           terminate_on_nan=True)
    propensity_treatment_trainer.fit(propensity_treatment)

    # Validation BCE
    val_bce_orig, val_bce_all = propensity_treatment.get_masked_bce(dataset_collection.val_f)
    logger.info(f'Val normalised BCE (all): {val_bce_all}; Val normalised RMSE (orig): {val_bce_orig}')

    # Test BCE
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual
        test_bce_orig, test_bce_all = propensity_treatment.get_masked_bce(dataset_collection.test_cf_one_step)
    elif hasattr(dataset_collection, 'test_f'):  # Test factual
        test_bce_orig, test_bce_all = propensity_treatment.get_masked_bce(dataset_collection.test_f)

    logger.info(f'Test normalised RMSE (all): {test_bce_orig}; Test normalised RMSE (orig): {test_bce_all}.')
    prop_treatment_results = {
        'propensity_treatment_val_bce_all': val_bce_all,
        'propensity_treatment_val_bce_orig': val_bce_orig,
        'propensity_treatment_test_bce_all': test_bce_all,
        'propensity_treatment_test_bce_orig': test_bce_orig
    }

    mlf_logger.log_metrics(prop_treatment_results) if args.exp.logging else None
    results.update(prop_treatment_results)

    # ============================== Denominator (history propensity network) ==============================
    propensity_history = instantiate(args.model.propensity_history, args, dataset_collection, _recursive_=False)
    if args.model.propensity_history.tune_hparams:
        propensity_history.finetune(resources_per_trial=args.model.propensity_history.resources_per_trial)

    propensity_history_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                         logger=mlf_logger,
                                         max_epochs=args.exp.max_epochs,
                                         callbacks=propensity_history_callbacks,
                                         gradient_clip_val=args.model.propensity_history.max_grad_norm,
                                         terminate_on_nan=True)
    propensity_history_trainer.fit(propensity_history)

    # Validation BCE
    val_bce_orig, val_bce_all = propensity_history.get_masked_bce(dataset_collection.val_f)
    logger.info(f'Val normalised BCE (all): {val_bce_all}; Val normalised RMSE (orig): {val_bce_orig}')

    # Test BCE
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual
        test_bce_orig, test_bce_all = propensity_history.get_masked_bce(dataset_collection.test_cf_one_step)
    elif hasattr(dataset_collection, 'test_f'):  # Test factual
        test_bce_orig, test_bce_all = propensity_history.get_masked_bce(dataset_collection.test_f)

    logger.info(f'Test normalised BCE (all): {test_bce_orig}; Test normalised BCE (orig): {test_bce_all}.')
    propensity_history_results = {
        'propensity_history_val_bce_all': val_bce_all,
        'propensity_history_val_bce_orig': val_bce_orig,
        'propensity_history_test_bce_all': test_bce_all,
        'propensity_history_test_bce_orig': test_bce_orig
    }

    mlf_logger.log_metrics(propensity_history_results) if args.exp.logging else None
    results.update(propensity_history_results)

    # ============================== Initialisation & Training of Encoder ==============================
    encoder = instantiate(args.model.encoder, args, propensity_treatment, propensity_history, dataset_collection,
                          _recursive_=False)
    if args.model.encoder.tune_hparams:
        encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

    encoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                              logger=mlf_logger,
                              max_epochs=args.exp.max_epochs,
                              callbacks=encoder_callbacks,
                              gradient_clip_val=args.model.encoder.max_grad_norm,
                              terminate_on_nan=True)
    encoder_trainer.fit(encoder)
    encoder_results = {}

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    encoder_trainer.test(encoder, test_dataloaders=val_dataloader)
    val_rmse_orig, val_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = encoder.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
                                                                                           one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    # ============================== Initialisation & Training of Decoder ==============================
    if args.model.train_decoder:
        decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)

        if args.model.decoder.tune_hparams:
            decoder.finetune(resources_per_trial=args.model.decoder.resources_per_trial)

        decoder_trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                                  logger=mlf_logger,
                                  max_epochs=args.exp.max_epochs,
                                  gradient_clip_val=args.model.decoder.max_grad_norm,
                                  callbacks=decoder_callbacks,
                                  terminate_on_nan=True)
        decoder_trainer.fit(decoder)

        # Validation factual rmse
        val_dataloader = DataLoader(dataset_collection.val_f, batch_size=10 * args.dataset.val_batch_size, shuffle=False)
        decoder_trainer.test(decoder, test_dataloaders=val_dataloader)
        val_rmse_orig, val_rmse_all = decoder.get_normalised_masked_rmse(dataset_collection.val_f)
        logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

        test_rmses = {}
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
            test_rmses = decoder.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, 'test_f'):  # Test n_step_factual rmse
            test_rmses = decoder.get_normalised_n_step_rmses(dataset_collection.test_f)
        test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
        decoder_results = {
            'decoder_val_rmse_all': val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig
        }
        decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

        mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
        results.update(decoder_results)

    return results


if __name__ == "__main__":
    main()

