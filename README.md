CausalTransformer
==============================
[![Conference](https://img.shields.io/badge/ICML22-Paper-blue])](https://proceedings.mlr.press/v162/melnychuk22a/melnychuk22a.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2204.07258-b31b1b.svg)](https://arxiv.org/abs/2204.07258)
[![Python application](https://github.com/Valentyn1997/CausalTransformer/actions/workflows/python-app.yml/badge.svg)](https://github.com/Valentyn1997/CausalTransformer/actions/workflows/python-app.yml)

Causal Transformer for estimating counterfactual outcomes over time.

<img width="1518" alt="Screenshot 2022-06-03 at 16 41 44" src="https://user-images.githubusercontent.com/23198776/171877145-c7cba15e-9787-4594-8f1f-cbb8b337b74a.png">


The project is built with following Python libraries:
1. [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - deep learning models
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [MlFlow](https://mlflow.org/) - experiments tracking

### Installations
First one needs to make the virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow Setup / Connection
To start an experiments server, run: 

`mlflow server --port=5000`

To access MlFLow web UI with all the experiments, connect via ssh:

`ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>`

Then, one can go to local browser http://localhost:5000.

## Experiments

Main training script is universal for different models and datasets. For details on mandatory arguments - see the main configuration file `config/config.yaml` and other files in `configs/` folder.

Generic script with logging and fixed random seed is following (with `training-type` `enc_dec`, `gnet`, `rmsn` and `multi`):
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_<training-type>.py +dataset=<dataset> +backbone=<backbone> exp.seed=10 exp.logging=True
```

### Backbones (baselines)
One needs to choose a backbone and then fill the specific hyperparameters (they are left blank in the configs):
- Causal Transformer (this paper): `runnables/train_multi.py  +backbone=ct`
- Encoder-Decoder Causal Transformer (this paper): `runnables/train_enc_dec.py  +backbone=edct`
- [Marginal Structural Models](https://pubmed.ncbi.nlm.nih.gov/10955408/) (MSMs): `runnables/train_msm.py +backbone=msm`
- [Recurrent Marginal Structural Networks](https://papers.nips.cc/paper/2018/hash/56e6a93212e4482d99c84a639d254b67-Abstract.html) (RMSNs): `runnables/train_rmsn.py +backbone=rmsn`
- [Counterfactual Recurrent Network](https://arxiv.org/abs/2002.04083) (CRN): `runnables/train_enc_dec.py +backbone=crn`
- [G-Net](https://proceedings.mlr.press/v158/li21a/li21a.pdf): `runnables/train_gnet.py  +backbone=gnet`


Models already have best hyperparameters saved (for each model and dataset), one can access them via: `+backbone/<backbone>_hparams/cancer_sim_<balancing_objective>=<coeff_value>` or `+backbone/<backbone>_hparams/mimic3_real=diastolic_blood_pressure`.

For CT, EDCT, and CT, several adversarial balancing objectives are available:
- counterfactual domain confusion loss (this paper): `exp.balancing=domain_confusion`
- gradient reversal (originally in CRN, but can be used for all the methods): `exp.balancing=grad_reverse`

To train a decoder (for CRN and RMSNs), use the flag `model.train_decoder=True`.

To perform a manual hyperparameter tuning use the flags `model.<sub_model>.tune_hparams=True`, and then see `model.<sub_model>.hparams_grid`. Use `model.<sub_model>.tune_range` to specify the number of trials for random search.


### Datasets
One needs to specify a dataset / dataset generator (and some additional parameters, e.g. set gamma for `cancer_sim` with `dataset.coeff=1.0`):
- Synthetic Tumor Growth Simulator: `+dataset=cancer_sim`
- MIMIC III Semi-synthetic Simulator (multiple treatments and outcomes): `+dataset=mimic3_synthetic`
- MIMIC III Real-world dataset: `+dataset=mimic3_real`

Before running MIMIC III experiments, place MIMIC-III-extract dataset ([all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract)) to `data/processed/`

Example of running Causal Transformer on Synthetic Tumor Growth Generator with gamma = [1.0, 2.0, 3.0] and different random seeds (total of 30 subruns), using hyperparameters:

```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +backbone/ct_hparams/cancer_sim_domain_conf=\'0\',\'1\',\'2\' exp.seed=10,101,1010,10101,101010
```

### Updated results

#### Self- and cross-attention bug
New results for semi-synthetic and real-world experiments after fixing a bug with self- and cross-attentions (https://github.com/Valentyn1997/CausalTransformer/issues/7). Therein, the bug affected only Tables 1 and 2, and Figure 5 (https://arxiv.org/pdf/2204.07258.pdf). Nevertheless, the performance of the CT with the bug fixed did not change drastically. 

*Table 1 (updated)*. Results for semi-synthetic data for $\tau$-step-ahead prediction based on real-world medical data (MIMIC-III). Shown: RMSE as mean ± standard deviation over five runs.

|                          | $\tau = 1$           | $\tau = 2$           | $\tau = 3$           | $\tau = 4$           | $\tau = 5$           | $\tau = 6$           | $\tau = 7$           | $\tau = 8$                      | $\tau = 9$                          | $\tau = 10$                         |
|:-------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|:--------------------------------|:------------------------------------|:------------------------------------|
| MSMs                     | 0.37 ± 0.01          | 0.57 ± 0.03          | 0.74 ± 0.06          | 0.88 ± 0.03          | 1.14 ± 0.10          | 1.95 ± 1.48          | 3.44 ± 4.57          | > 10.0 | > 10.0 | > 10.0 |
| RMSNs                    | 0.24 ± 0.01          | 0.47 ± 0.01          | 0.60 ± 0.01          | 0.70 ± 0.02          | 0.78 ± 0.04          | 0.84 ± 0.05          | 0.89 ± 0.06          | 0.94 ± 0.08                     | 0.97 ± 0.09                         | 1.00 ± 0.11                         |
| CRN                      | 0.30 ± 0.01          | 0.48 ± 0.02          | 0.59 ± 0.02          | 0.65 ± 0.02          | 0.68 ± 0.02          | 0.71 ± 0.01          | 0.72 ± 0.01          | 0.74 ± 0.01                     | 0.76 ± 0.01                         | 0.78 ± 0.02                         |
| G-Net                    | 0.34 ± 0.01          | 0.67 ± 0.03          | 0.83 ± 0.04          | 0.94 ± 0.04          | 1.03 ± 0.05          | 1.10 ± 0.05          | 1.16 ± 0.05          | 1.21 ± 0.06                     | 1.25 ± 0.06                         | 1.29 ± 0.06                         |
| EDCT (GR; $\lambda = 1$) | 0.29 ± 0.01          | 0.46 ± 0.01          | 0.56 ± 0.01          | 0.62 ± 0.01          | 0.67 ± 0.01          | 0.70 ± 0.01          | 0.72 ± 0.01          | 0.74 ± 0.01                     | 0.76 ± 0.01                         | 0.78 ± 0.01                         |
| CT ($\alpha = 0$) (ours, fixed) | **0.20 ± 0.01** | **0.38 ± 0.01** | **0.46 ± 0.01** | **0.50 ± 0.01** | **0.52 ± 0.01** | **0.54 ± 0.01** | 0.56 ± 0.01          | **0.57 ± 0.01**            | 0.59 ± 0.01                         | 0.60 ± 0.01                         |
| CT (ours, fixed)                | 0.21 ± 0.01          | **0.38 ± 0.01** | **0.46 ± 0.01** | **0.50 ± 0.01** | 0.53 ± 0.01          | **0.54 ± 0.01** | **0.55 ± 0.01** | **0.57 ± 0.01**            | **0.58 ± 0.01**                | **0.59 ± 0.01**                |

*Table 2 (updated)*. Results for experiments with real-world medical data (MIMIC-III). Shown: RMSE as mean ± standard deviation over five runs.

|           | $\tau = 1$           | $\tau = 2$           | $\tau = 3$           | $\tau = 4$           | $\tau = 5$            |
|:----------|:---------------------|:---------------------|:---------------------|:---------------------|:----------------------|
| MSMs      | 6.37 ± 0.26          | 9.06 ± 0.41          | 11.89 ± 1.28         | 13.12 ± 1.25         | 14.44 ± 1.12          |
| RMSNs     | 5.20 ± 0.15          | 9.79 ± 0.31          | 10.52 ± 0.39         | 11.09 ± 0.49         | 11.64 ± 0.62          |
| CRN       | 4.84 ± 0.08          | 9.15 ± 0.16          | 9.81 ± 0.17          | 10.15 ± 0.19         | 10.40 ± 0.21          |
| G-Net     | 5.13 ± 0.05          | 11.88 ± 0.20         | 12.91 ± 0.26         | 13.57 ± 0.30         | 14.08 ± 0.31          |
| CT (ours, fixed) | **4.60 ± 0.08**      | **9.01 ± 0.21**      | **9.58 ± 0.19**      | **9.89 ± 0.21**      | **10.12 ± 0.22**      |

*Figure 6 (updated)*. Subnetworks importance scores based on semi-synthetic benchmark (higher values correspond to higher importance of subnetwork connectivity via cross-attentions). Shown: RMSE differences between model with isolated subnetwork and full CT, means ± standard errors.

![subnet-isolation](https://github.com/Valentyn1997/CausalTransformer/assets/23198776/a05925fc-bc38-488a-ac23-bab5d79892e2)

#### Last active entry zeroing bug

New results after fixing a bug with the synthetic tumor-growth simulator: outcome corresponding to the last entry for every time series was zeroed.

*Table 9 (updated)*. Normalized RMSE for one-step-ahead prediction. Shown: mean and standard deviation over five runs (lower is better). Parameter $\gamma$ is the the amount of time-varying confounding: higher values mean larger treatment assignment bias.

|                          | $\gamma = 0$      | $\gamma = 1$      | $\gamma = 2$      | $\gamma = 3$      | $\gamma = 4$      |
|:-------------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| MSMs                     | 1.091 ± 0.115     | 1.202 ± 0.108     | 1.383 ± 0.090     | 1.647 ± 0.121     | 1.981 ± 0.232     |
| RMSNs                    | 0.834 ± 0.072     | 0.860 ± 0.025     | 1.000 ± 0.134     | 1.131 ± 0.057     | 1.434 ± 0.148     |
| CRN                      | **0.755 ± 0.059** | 0.788 ± 0.057     | 0.881 ± 0.066     | 1.062 ± 0.088     | 1.358 ± 0.167     |
| G-Net                    | 0.795 ± 0.066     | 0.841 ± 0.038     | 0.946 ± 0.083     | **1.057 ± 0.146** | **1.319 ± 0.248** |
| CT ($\alpha = 0$) (ours) | 0.772 ± 0.051     | **0.783 ± 0.071** | **0.862 ± 0.052** | 1.062 ± 0.119     | 1.331 ± 0.217     |
| CT (ours)                | 0.770 ± 0.049     | **0.783 ± 0.071** | 0.864 ± 0.059     | 1.098 ± 0.097     | 1.413 ± 0.259     |


*Table 10 (updated)*. Normalized RMSE for $\tau$-step-ahead prediction (here: random trajectories setting). Shown: mean and standard deviation over five runs (lower is better). Parameter $\gamma$ is the amount of time-varying confounding: higher values mean larger treatment assignment bias.

|                                    | $\gamma = 0$      | $\gamma = 1$      | $\gamma = 2$      | $\gamma = 3$      | $\gamma = 4$      |
|:-----------------------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| ('2', 'MSMs')                      | 0.975 ± 0.063     | 1.183 ± 0.146     | 1.428 ± 0.274     | 1.673 ± 0.431     | 1.884 ± 0.637     |
| ('2', 'RMSNs')                     | 0.825 ± 0.057     | 0.851 ± 0.043     | 0.861 ± 0.078     | 0.993 ± 0.126     | 1.269 ± 0.294     |
| ('2', 'CRN')                       | **0.761 ± 0.058** | **0.760 ± 0.037** | **0.805 ± 0.050** | 2.045 ± 1.491     | 1.209 ± 0.192     |
| ('2', 'G-Net')                     | 1.006 ± 0.082     | 0.994 ± 0.086     | 1.185 ± 0.077     | 1.083 ± 0.145     | 1.243 ± 0.202     |
| ('2', 'CT ($\\alpha = 0$) (ours)') | 0.766 ± 0.029     | 0.781 ± 0.066     | 0.814 ± 0.078     | **0.944 ± 0.144** | 1.191 ± 0.316     |
| ('2', 'CT (ours)')                 | 0.762 ± 0.028     | 0.781 ± 0.058     | 0.818 ± 0.091     | 1.001 ± 0.150     | **1.163 ± 0.233** |
| ('3', 'MSMs')                      | 0.937 ± 0.060     | 1.133 ± 0.158     | 1.344 ± 0.262     | 1.525 ± 0.400     | 1.564 ± 0.545     |
| ('3', 'RMSNs')                     | 0.824 ± 0.043     | 0.871 ± 0.036     | 0.857 ± 0.109     | 1.020 ± 0.140     | **1.267 ± 0.298** |
| ('3', 'CRN')                       | 0.769 ± 0.057     | **0.777 ± 0.037** | **0.826 ± 0.077** | 1.789 ± 1.108     | 1.356 ± 0.330     |
| ('3', 'G-Net')                     | 1.103 ± 0.092     | 1.097 ± 0.095     | 1.355 ± 0.107     | 1.225 ± 0.184     | 1.382 ± 0.242     |
| ('3', 'CT ($\\alpha = 0$) (ours)') | 0.766 ± 0.037     | 0.806 ± 0.060     | 0.828 ± 0.106     | **0.996 ± 0.185** | 1.335 ± 0.465     |
| ('3', 'CT (ours)')                 | **0.762 ± 0.036** | 0.807 ± 0.056     | 0.838 ± 0.120     | 1.072 ± 0.196     | 1.283 ± 0.312     |
| ('4', 'MSMs')                      | 0.845 ± 0.060     | 1.022 ± 0.149     | 1.196 ± 0.233     | 1.325 ± 0.363     | 1.308 ± 0.482     |
| ('4', 'RMSNs')                     | 0.780 ± 0.046     | 0.834 ± 0.040     | 0.814 ± 0.123     | 0.988 ± 0.146     | **1.169 ± 0.269** |
| ('4', 'CRN')                       | 0.734 ± 0.061     | **0.743 ± 0.037** | 0.805 ± 0.096     | 1.567 ± 0.825     | 1.327 ± 0.293     |
| ('4', 'G-Net')                     | 1.092 ± 0.090     | 1.074 ± 0.098     | 1.385 ± 0.117     | 1.212 ± 0.202     | 1.358 ± 0.253     |
| ('4', 'CT ($\\alpha = 0$) (ours)') | 0.730 ± 0.042     | 0.776 ± 0.056     | **0.802 ± 0.119** | **0.983 ± 0.208** | 1.394 ± 0.563     |
| ('4', 'CT (ours)')                 | **0.726 ± 0.041** | 0.777 ± 0.054     | 0.810 ± 0.128     | 1.075 ± 0.220     | 1.302 ± 0.356     |
| ('5', 'MSMs')                      | 0.747 ± 0.056     | 0.896 ± 0.136     | 1.038 ± 0.210     | 1.128 ± 0.320     | 1.155 ± 0.448     |
| ('5', 'RMSNs')                     | 0.717 ± 0.053     | 0.775 ± 0.041     | **0.747 ± 0.124** | **0.922 ± 0.141** | **1.057 ± 0.246** |
| ('5', 'CRN')                       | 0.678 ± 0.062     | **0.692 ± 0.037** | 0.761 ± 0.104     | 1.410 ± 0.604     | 1.242 ± 0.239     |
| ('5', 'G-Net')                     | 1.033 ± 0.086     | 1.014 ± 0.097     | 1.358 ± 0.118     | 1.160 ± 0.199     | 1.285 ± 0.242     |
| ('5', 'CT ($\\alpha = 0$) (ours)') | 0.673 ± 0.044     | 0.722 ± 0.052     | 0.748 ± 0.124     | 0.931 ± 0.213     | 1.405 ± 0.648     |
| ('5', 'CT (ours)')                 | **0.669 ± 0.043** | 0.723 ± 0.053     | 0.751 ± 0.125     | 1.036 ± 0.238     | 1.264 ± 0.389     |
| ('6', 'MSMs')                      | 0.647 ± 0.055     | 0.778 ± 0.123     | 0.894 ± 0.188     | 0.952 ± 0.284     | 1.060 ± 0.432     |
| ('6', 'RMSNs')                     | 0.646 ± 0.058     | 0.702 ± 0.043     | **0.675 ± 0.121** | **0.847 ± 0.132** | **0.947 ± 0.225** |
| ('6', 'CRN')                       | 0.614 ± 0.057     | **0.631 ± 0.035** | 0.706 ± 0.104     | 1.308 ± 0.438     | 1.132 ± 0.194     |
| ('6', 'G-Net')                     | 0.963 ± 0.083     | 0.942 ± 0.090     | 1.321 ± 0.118     | 1.092 ± 0.183     | 1.195 ± 0.223     |
| ('6', 'CT ($\\alpha = 0$) (ours)') | 0.609 ± 0.042     | 0.657 ± 0.046     | 0.684 ± 0.122     | 0.864 ± 0.201     | 1.383 ± 0.699     |
| ('6', 'CT (ours)')                 | **0.605 ± 0.040** | 0.657 ± 0.047     | 0.685 ± 0.119     | 0.979 ± 0.249     | 1.201 ± 0.419     |


*Table 11 (updated)*. Normalized RMSE for $\tau$-step-ahead prediction (here: single sliding treatment setting). Shown: mean and standard deviation over five runs (lower is better). Parameter $\gamma$ is the amount of time-varying confounding: higher values mean larger treatment assignment bias.

|                                    | $\gamma = 0$      | $\gamma = 1$      | $\gamma = 2$      | $\gamma = 3$      | $\gamma = 4$      |
|:-----------------------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| ('2', 'MSMs')                      | 1.362 ± 0.109     | 1.612 ± 0.172     | 1.939 ± 0.365     | 2.290 ± 0.545     | 2.468 ± 1.058     |
| ('2', 'RMSNs')                     | 0.742 ± 0.043     | 0.760 ± 0.047     | 0.827 ± 0.056     | 0.957 ± 0.106     | 1.276 ± 0.240     |
| ('2', 'CRN')                       | **0.671 ± 0.066** | **0.666 ± 0.052** | 0.741 ± 0.042     | 1.668 ± 1.184     | 1.151 ± 0.166     |
| ('2', 'G-Net')                     | 1.021 ± 0.067     | 1.009 ± 0.092     | 1.271 ± 0.075     | 1.113 ± 0.149     | 1.257 ± 0.227     |
| ('2', 'CT ($\\alpha = 0$) (ours)') | 0.685 ± 0.050     | 0.679 ± 0.044     | 0.714 ± 0.053     | **0.875 ± 0.105** | **1.072 ± 0.315** |
| ('2', 'CT (ours)')                 | 0.681 ± 0.052     | 0.677 ± 0.044     | **0.713 ± 0.042** | 0.908 ± 0.122     | 1.274 ± 0.366     |
| ('3', 'MSMs')                      | 1.679 ± 0.132     | 1.953 ± 0.208     | 2.302 ± 0.437     | 2.640 ± 0.639     | 2.622 ± 1.132     |
| ('3', 'RMSNs')                     | 0.783 ± 0.053     | 0.792 ± 0.047     | 0.889 ± 0.050     | 1.086 ± 0.175     | 1.382 ± 0.286     |
| ('3', 'CRN')                       | **0.700 ± 0.078** | **0.692 ± 0.046** | 0.818 ± 0.051     | 1.959 ± 1.032     | 1.360 ± 0.225     |
| ('3', 'G-Net')                     | 1.253 ± 0.079     | 1.226 ± 0.104     | 1.611 ± 0.102     | 1.383 ± 0.200     | 1.574 ± 0.328     |
| ('3', 'CT ($\\alpha = 0$) (ours)') | 0.707 ± 0.053     | 0.711 ± 0.038     | **0.770 ± 0.043** | **0.969 ± 0.119** | **1.261 ± 0.462** |
| ('3', 'CT (ours)')                 | 0.703 ± 0.055     | 0.712 ± 0.040     | **0.770 ± 0.032** | 1.010 ± 0.119     | 1.536 ± 0.450     |
| ('4', 'MSMs')                      | 1.871 ± 0.145     | 2.145 ± 0.227     | 2.489 ± 0.471     | 2.791 ± 0.681     | 2.615 ± 1.142     |
| ('4', 'RMSNs')                     | 0.821 ± 0.079     | 0.837 ± 0.058     | 0.963 ± 0.106     | 1.216 ± 0.240     | **1.416 ± 0.304** |
| ('4', 'CRN')                       | 0.734 ± 0.087     | **0.722 ± 0.041** | 0.898 ± 0.068     | 2.201 ± 0.967     | 1.573 ± 0.255     |
| ('4', 'G-Net')                     | 1.390 ± 0.087     | 1.347 ± 0.112     | 1.819 ± 0.133     | 1.544 ± 0.243     | 1.769 ± 0.413     |
| ('4', 'CT ($\\alpha = 0$) (ours)') | 0.729 ± 0.056     | 0.749 ± 0.033     | 0.826 ± 0.046     | **1.053 ± 0.147** | 1.426 ± 0.574     |
| ('4', 'CT (ours)')                 | **0.726 ± 0.057** | 0.748 ± 0.036     | **0.822 ± 0.036** | 1.089 ± 0.122     | 1.762 ± 0.523     |
| ('5', 'MSMs')                      | 1.963 ± 0.155     | 2.221 ± 0.231     | 2.547 ± 0.479     | 2.810 ± 0.684     | 2.542 ± 1.122     |
| ('5', 'RMSNs')                     | 0.855 ± 0.099     | 0.889 ± 0.074     | 1.030 ± 0.165     | 1.349 ± 0.326     | **1.434 ± 0.299** |
| ('5', 'CRN')                       | 0.769 ± 0.094     | **0.755 ± 0.039** | 0.976 ± 0.082     | 2.361 ± 1.000     | 1.730 ± 0.292     |
| ('5', 'G-Net')                     | 1.477 ± 0.092     | 1.430 ± 0.119     | 1.963 ± 0.157     | 1.667 ± 0.275     | 1.907 ± 0.471     |
| ('5', 'CT ($\\alpha = 0$) (ours)') | 0.758 ± 0.055     | 0.788 ± 0.036     | 0.875 ± 0.056     | **1.118 ± 0.172** | 1.560 ± 0.663     |
| ('5', 'CT (ours)')                 | **0.756 ± 0.057** | 0.786 ± 0.039     | **0.870 ± 0.048** | 1.154 ± 0.111     | 1.922 ± 0.569     |
| ('6', 'MSMs')                      | 1.970 ± 0.155     | 2.205 ± 0.228     | 2.509 ± 0.469     | 2.732 ± 0.662     | 2.422 ± 1.084     |
| ('6', 'RMSNs')                     | 0.889 ± 0.112     | 0.936 ± 0.091     | 1.081 ± 0.211     | 1.473 ± 0.433     | **1.436 ± 0.290** |
| ('6', 'CRN')                       | 0.807 ± 0.097     | **0.790 ± 0.035** | 1.047 ± 0.092     | 2.480 ± 1.078     | 1.827 ± 0.326     |
| ('6', 'G-Net')                     | 1.538 ± 0.091     | 1.493 ± 0.121     | 2.062 ± 0.172     | 1.758 ± 0.286     | 1.994 ± 0.500     |
| ('6', 'CT ($\\alpha = 0$) (ours)') | 0.790 ± 0.058     | 0.827 ± 0.036     | 0.915 ± 0.063     | **1.177 ± 0.193** | 1.654 ± 0.704     |
| ('6', 'CT (ours)')                 | **0.789 ± 0.059** | 0.821 ± 0.034     | **0.909 ± 0.054** | 1.205 ± 0.100     | 2.052 ± 0.608     |

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
