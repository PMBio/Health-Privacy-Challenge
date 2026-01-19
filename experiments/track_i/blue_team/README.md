# :blueberries: BLUE TEAM 

Welcome to your home page! 

**Contents**
- [Prerequisites](#prerequisites)
- [Tasks](#tasks)
- [Guideline for running and evaluating baseline methods](#thread-guideline-for-running-and-evaluating-baseline-methods)
- [Guideline for developing your own method](#bookmark_tabs-guideline-for-developing-your-own-method)
- [Submission checklist](#white_check_mark-submission-checklist)


## Prerequisites

### Python environment

We suggest installing the provided environments. We recommend using micromamba, though you may also replace mamba with conda. 

```bash
micromamba create --file environment.yaml --name <env_name>
```

To be able to run CTGAN with Differential Privacy we utilize ``smartnoise-synth`` package which requires a specific version of ``opacus``. Thus, we provide a specific environment to be able to reproduce this experiment. 

```bash
micromamba create --file dpctgan_environment.yaml --name <env_name>
```


## Tasks 
- Blue teams work towards developing methods that improve **the baseline methods and generating novel insights into privacy preservation** in biological datasets. 
  - Please review the [CAMDA 2025 Challenge](https://benchmarks.elsa-ai.eu/?ch=4&com=evaluation&task=1) evaluation board to examine the participant submissions. 
- Blue teams will use the provided scripts to generate a  **stratified five-fold dataset splits** and the performance of their methods will be evaluated on the average performance. 
- Each blue team must use a **unique random seed** to generate their dataset splits and run their method.
  - For example, in the provided examples, value of 42, is assigned to the random seed, in both dataset and generator method configurations.
  - Blue teams must run their experiments with a unique random seed consistent throughout `dataset_config` and `{generator_name}_config` configurations in the `config.yaml` file. 
 

## :thread: Guideline for running and evaluating baseline methods

Please follow the three main steps below:
1. **[Create cross-fold dataset splits](/experiments/track_i/blue_team/1_data/)**: Prepares the data in stratified 5 fold train-test splits. 
2. **[Generate synthetic data](/experiments/track_i/blue_team/2_generation/)**: Generates synthetic data using baseline methods on each data split generated in the previous step. 
3. **[Evaluate](/experiments/track_i/blue_team/3_evaluation/)**: Reports the evaluation metrics for each data splits (and in average).


## :bookmark_tabs: Guideline for developing your own method

We provide `BaseDataGenerator` class inside [generators/models/base.py](/src/generators/models/base.py) for generator models to inherit. This class handles the configuration arguments passed through `config.yaml`. 

It also has three abstract functions:  
- `train()`, 
- `load_from_checkpoint()`, 
- `generate()`, 
-  and `save_synthetic_data()`, a fixed function that implements saving the synthetic datasets in the desired format. 

We expect your submitted code to follow the logic in [blue_team.py](/src/generators/blue_team.py). Your generate() function should return synthetic data and synthetic labels.

```bash
@click.command()
@click.argument('split_no', type=int)
@click.option('--experiment_name', type=str, default="")
def run_generator(split_no: int, experiment_name: str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)

    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config, split_no=split_no)

    if not isinstance(generator, MultivariateDataGenerator):
        if not config.get("load_from_checkpoint", False):
            if config.get("train", False):
                generator.train()
        else:
            generator.load_from_checkpoint()

    if config.get("generate", True):
        syn_data, syn_lbl = generator.generate()
        generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)
```
Please test your code inside `run_generator` function to ensure its reproducibility. 


## :white_check_mark: Submission checklist
The following files are required for submission, for each dataset:

1. Modified [blue_team.py](/src/generators/blue_team.py) with your generator class included, and any additional `.py` file required to run your method. 
2.  Five sets of synthetic data and labels saved with `save_synthetic_data()` functionality
3.  Updated `config.yaml` file containing `{your_generator}_config` and `random_seed` with a new value inside `dataset_config` and `{your_generator}_config`. 
    - Please note that a new value for random seed means that: **each blue team has different real and synthetic datasets based on the split using this value.** 
4. `{dataset_name}_split.yaml` generated with the new `random_seed` value inside dataset_config.
5. `environment.yaml` that is required to run your method.


This means we expect two zip files from you in the following file name format,
1.  `blueteam_{yourteamname}_TCGA-BRCA.zip`
2.  `blueteam_{yourteamname}_TCGA-COMBINED.zip`


Please take look at the [example submission zip file](/experiments/track_i/blue_team/blueteam_example_TCGA-BRCA.zip) to double check correct formatting. 


**Best of luck!** :four_leaf_clover:





