# General information on dataset formats

To run custom datasets with Causal Transformer or other time-varying treatment effect baselines, one needs to define two classes:
- `CustomDataset` class, inherited from `torch.utils.data.Dataset`
- `CustomDatasetCollection` class, inherited from `src.data.dataset_collection.RealDatasetCollection` or `src.data.dataset_collection.SyntheticDatasetCollection`. It is composed of several `CustomDataset` classes corresponding to (factual) train and validation subsets, and different counterfactual or factual test subsets.

#### `CustomDataset` 
To create a custom dataset, one needs to create its own `CustomDataset` class. It has the same basic structure for all the datasets with the following initialization:

```python
class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = ... # dictinary with unstructured time-series data

        self.processed = False  # Flag for turning the unstructured data to structured data
        self.processed_sequential = False  # Flag for turning the structured data to structured exploaded data
        self.processed_autoregressive = False  # Flag for creating a copy of the structured exploaded data for the several-step-ahead prediction
        self.treatment_mode = treatment_mode  # multiclass / multilabel
        self.exploded = False  # Flag, whether the data is exploaded

        self.norm_const  # Normalization constant for the 
        ...
```

Then, the following methods are used to process the data for different stages of learning / evaluation:
- `process_data(scaling_params)` sets `self.processed = True` and converts the unstructured `self.data` to the structured `self.data` with the following keys:
  ```python
  self.data = {
      'sequence_lengths': ...,  # Sequence lengths
      'prev_treatments': ...,  # Past treatment, part of history, $\bar{H}_t$
      'vitals': ..., # Current time-varying covariates, part of history, $\bar{H}_t$
      'prev_outputs': ...,  # Previous outcomes, part of history, $\bar{H}_t$ 
      'static_features': ..., # Static covariates, Part of history, $\bar{H}_t$ 
      'current_treatments': ..., # Current treatment (= one-time-step shifted prev_treatments), $A_t$
      'outputs': ..., # Current outcome (= one-time-step shifted prev_outputs), $Y_t$
      'next_vitals': ..., # Next time-varying covariates, $X_{t+1}$
      'active_entries': ..., # Mask with sequence lengths
      'scaling_params': ... # Normalization parameters  
  }
  ```
- `explode_trajectories(projection_horizon)` cuts time-series in `self.data` in chunks with the rolling prediction origin.
- `process_sequential(encoder_r, projection_horizon)` sets `self.processed_sequential = True` and prepares `self.data` to be used for training for multi-step-ahead prediction
- `process_sequential_test(self, projection_horizon)` sets `self.processed_sequential = True` and prepares `self.data` to be used for evaluation for multi-step-ahead prediction
- `process_autoregressive_test(self, encoder_r, encoder_outputs, projection_horizon)` sets `self.processed_autoregressive = True` and prepares `self.data` for the auto-regressive multi-step-ahead prediction
- `process_sequential_multi(projection_horizon)` sets `self.processed_autoregressive = True` and prepares `self.data` for the multi-step-ahead prediction with the Causal Transformer


#### `CustomDatasetCollection`

Furthermore, to use own dataset, one needs to create `CustomDatasetCollection` class, inherited from `src.data.dataset_collection.RealDatasetCollection` or `src.data.dataset_collection.SyntheticDatasetCollection`, where **only the initialization has to be overwritten&&. `CustomDatasetCollection` combines several `CustomDataset` instances, corresponding to the train, validation and different test subsets. It contains different ready-to-use methods for pre-processing the data, e.g., `process_data_encoder()`, `process_data_decoder()`, `process_data_multi()`, etc. See the Table below with different stages of pre-processing and training:

| Model              | One-step-ahead prediction                                                                                                                                                               | Multi-step-ahead prediction                                                                    |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| MSMs               | 1. `process_data_multi()` <br> 2. fit 2 logistic regressions for IPTW <br>  3. `process_propensity_train_f()` <br> 4. fit linear regressions <br> 5. evaluate one-step-ahead prediction | 6. evaluate multi-step-ahead prediction                                                        |
| RMSNs              | 1. `process_data_encoder()` <br> 2. fit 2 networks for IPTW <br> 3. `process_propensity_train_f()` <br> 4. fit an encoder <br> 5. evaluate one-step-ahead prediction                    | 6. `process_data_decoder()` <br> 7. fit a decoder <br> 8. evaluate multi-step-ahead prediction |
| CRN                | 1. `process_data_encoder()` <br> 2. fit an encoder <br> 3. evaluate one-step-ahead prediction                                                                                           | 4. `process_data_decoder()` <br> 5. fit a decoder <br> 6. evaluate multi-step-ahead prediction |
| G-Net              | 1. `process_data_multi()` <br> 2. `split_train_f_holdout()` <br> 3. fit a single network <br> 4. evaluate one-step-ahead prediction                                                     | 5. `explode_cf_treatment_seq()` <br> 6. evaluate multi-step-ahead prediction                   |
| Causal Transformer | 1. `process_data_multi()` <br> 2. fit the multi-input network <br> 3. evaluate one-step-ahead prediction                                                                                | 4. evaluate multi-step-ahead prediction                                                        |