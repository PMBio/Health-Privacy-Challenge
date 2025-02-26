import click
import yaml
import os
import sys
#import torch
import importlib

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.utils.prepare_data import RealDataLoader
from generators.models.multivariate import MultivariateDataGenerator

generator_classes = {
    'multivariate': ('models.multivariate', 'MultivariateDataGenerator'),
    'cvae': ('models.cvae', 'CVAEDataGenerationPipeline'),
    'dpcvae': ('models.cvae', 'CVAEDataGenerationPipeline'),
    "ctgan": ('models.sdv_ctgan', 'CTGANDataGenerationPipeline'),
    "dpctgan": ('models.dpctgan', 'DPCTGANDataGenerationPipeline'),
    "sc_dist": ('models.sc_dist', 'ScDistributionDataGenerator'),
    "sc_dist_sparse": ('models.sc_dist_opt', 'ScDistributionDataGenerator')
}

## dynamic import to avoid package versioning errors
def get_generator_class(generator_name):
    if generator_name in generator_classes:
        module_name, class_name = generator_classes[generator_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Unknown generator name: {generator_name}")



@click.group()
def cli():
    pass

## a stratified 5 fold CV split will be created under
## data_splits/split_indices/{dataset_name}_split.yaml
## update random_seed in dataset_config to generate an original split
@click.command()
def generate_split_indices():
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))
    rdataloader = RealDataLoader(config)
    rdataloader.save_split_indices()

## the real data will be split into 5 train/test pairs
## based on the above generated {dataset_name}_split.yaml
## the data will be saved under data_splits/{dataset_name}/real/
@click.command()
def generate_data_splits():
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))
    rdataloader = RealDataLoader(config)
    # Save dataset
    rdataloader.save_split_data()


## your synthetic data will be saved accordingly to config.yaml
## e.g. data_splits/{dataset_name}/synthetic/{generator_name}/{experiment_name}
## change the corresponding keys in the config.yaml
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



## your synthetic data will be saved accordingly to config.yaml
## e.g. data_splits/{dataset_name}/synthetic/{generator_name}/{experiment_name}
## change the corresponding keys in the config.yaml
@click.command()
@click.option('--experiment_name', type=str, default="")
def run_singlecell_generator(experiment_name: str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)


    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config)


    if not config.get("load_from_checkpoint", False):
        if config.get("train", False):
            generator.train()
    else:
        generator.load_from_checkpoint()

    if config.get("generate", False):
        syn_data = generator.generate()
        generator.save_synthetic_anndata(syn_data, experiment_name)





cli.add_command(generate_data_splits)
cli.add_command(generate_split_indices)
cli.add_command(run_generator)
cli.add_command(run_singlecell_generator)
if __name__ == '__main__':
    cli()






# Check if CUDA is available
#def check_cuda_availability():
#    cuda_available = torch.cuda.is_available()
#    if cuda_available:
#        print("CUDA is available.")
#   else:
#        print("CUDA is NOT available.")

#check_cuda_availability()
