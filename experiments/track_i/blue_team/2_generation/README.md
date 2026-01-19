# :blueberries: BLUE TEAM: Generate synthetic bulk RNA-seq data 

## Prerequisites

You need to modify `config.yaml` according to your need for each experiment. Remember to update `home_dir`. The other configurations remain as they are depending on your preferences. 

### Activate the environment

```bash
micromamba activate <environment>
```

This section assumes you already generated real data splits. Please refer to [data](/experiments/track_i/blue_team/1_data/) if not. 


## Generate synthetic data with baseline methods for each split
The below code generate synthetic data for each real data split `split_1, split_2` etc., generated previously, where `$split_num` is the number of splits defined in config.yaml. It's set as follows: `split_num: 5`. **Please keep it as it is.** 

We provide multiple baseline methods, presented in total seven configurations: 

- `multivariate`: Multivariate normal sampling from average gene expression levels per subtype/type.
- `cvae` & `dpcvae`: Conditional Variational Autoencoder (CVAE, Sohn et al., 2015) without and **with Differential Privacy (DP)**, respectively. 
- `ctgan` & `dpctgan`:Conditional Generative Adversarial Networks (CTGAN, Xu et al., 2019) without and **with Differential Privacy (DP)**, respectively.
- `cvae_gmm`: CVAE with Gaussian Mixture Models to replaces the single Gaussian prior (Apellániz et al. 2024)
- `wgan_gp`: Wasserstein GAN with Gradient Penalty (Arjovsky et al., 2017, Gulrajani et al., 2017). 

In order to run these methods, you first need to modify inside the ``config.yaml``. Each generator model has their own configuration arguments. e.g. ``{generator_name}_config``. 

 ### Example 1: Multivariate

The first example in `config.yaml` uses Multivariate Normal distribution and adds multivariate noise based on the `noise level` specified in ``multivariate_config``.

Multivariate method requires  ``--experiment_name {experiment_name}`` argument to be set. And example experiment_name is, ``--experiment_name "noise_0.5"`` indicating generator configuration.

```bash
generator_name: "multivariate"
generate: True
train: False
load_from_checkpoint: False

multivariate_config:
  noise_level: 0.5


```

Then simply run, 

```bash
 python {src_dir}/generators/blue_team.py run-generator {split_no} --experiment_name {experiment_name}
```

 This will generate the data for the corresponding split (split_1, split_2, etc.), under data_splits directory,

 ``data_splits/{dataset_name}/synthetic/{generator_name}/{experiment_name}/synthetic_data_split_{split_num}.csv``

Make sure to run the script per each  `split_no` to complete the experiment. 



 ### Example 2: CVAE with Differential Privacy

To run the conditional VAE baselines, open the `config.yaml`  inside this directoy and modify the `generator_name`. You can then modify `cvae_config` parameters according to your needs. 

Experiment name is automatically set here, and in the rest of the generator methods. 


 ```bash
generator_name: "cvae"
generate: True
train: True
load_from_checkpoint: False

cvae_config: 
  device_type: "cuda"
  random_seed: 42 # seed experiment for reproducibility
  preprocess: "standard"

  train_config:
    num_iters: 10000
    batch_size: 64
  .
  .
  .
```
Then simply run, 

```bash
 python {src_dir}/generators/blue_team.py run-generator {split_no} 
```


In order to run CVAE with DP setting, modify the `generator_name: "dpcvae"`.
