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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for CT (Causal Transformer)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Initialisation of data
    seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    multimodel_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        multimodel_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== Initialisation & Training of multimodel ==============================
    multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
    if args.model.multi.tune_hparams:
        multimodel.finetune(resources_per_trial=args.model.multi.resources_per_trial)

    multimodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                 callbacks=multimodel_callbacks, terminate_on_nan=True,
                                 gradient_clip_val=args.model.multi.max_grad_norm)
    multimodel_trainer.fit(multimodel)

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    multimodel_trainer.test(multimodel, test_dataloaders=val_dataloader)
    # multimodel.visualize(dataset_collection.val_f, index=0, artifacts_path=artifacts_path)
    val_rmse_orig, val_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = multimodel.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
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
        test_rmse_orig, test_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.test_f)
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

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
    test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    decoder_results = {
        'decoder_val_rmse_all': val_rmse_all,
        'decoder_val_rmse_orig': val_rmse_orig
    }
    decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

    mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
    results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()

