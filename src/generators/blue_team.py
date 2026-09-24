import click
import yaml
import os
import sys
import torch
import importlib

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

from generators.utils.prepare_data import RealDataLoader
from generators.models.multivariate import MultivariateDataGenerator
from generators.utils.resource import ResourceProfiler



generator_classes = {
    'multivariate': ('models.multivariate', 'MultivariateDataGenerator'),
    'cvae': ('models.cvae', 'CVAEDataGenerationPipeline'),
    'dpcvae': ('models.cvae', 'CVAEDataGenerationPipeline'),
    'ctgan': ('models.sdv_ctgan', 'CTGANDataGenerationPipeline'),
    'dpctgan': ('models.dpctgan', 'DPCTGANDataGenerationPipeline'),
    'sc_dist': ('models.sc_dist', 'ScDistributionDataGenerator'),
    'cvae_gmm': ('models.cvae_gmm', 'CVAEGMMDataGenerator'),
    'wgan_gp': ('models.wgan_gp', 'WGANGPDataGenerator')
}

## dynamic import to avoid package versioning errors 
def get_generator_class(generator_name):
    if generator_name in generator_classes:
        module_name, class_name = generator_classes[generator_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Unknown generator name: {generator_name}")


def _get_module_for_params(generator):
    """Return the nn.Module(s) to count params from, as a list."""
    # CVAEDataGenerationPipeline
    if hasattr(generator, "model"):
        m = generator.model
        # CTGAN wraps the real nn.Module inside ._generator
        if not hasattr(m, "parameters") and hasattr(m, "_generator"):
            return [m._generator]
        if hasattr(m, "parameters"):
            return [m]
    # CVAEGMMDataGenerator
    if hasattr(generator, "cvae") and generator.cvae is not None:
        return [generator.cvae]
    # WGANGPDataGenerator (count both gen + disc)
    if hasattr(generator, "generator") and generator.generator is not None:
        mods = [generator.generator]
        if hasattr(generator, "discriminator") and generator.discriminator is not None:
            mods.append(generator.discriminator)
        return mods
    return []


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
def run_generator(split_no: int, experiment_name: str = None): #come back to this later 
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



##### RESOURCE CONSUMPTION SUMMARY TABLE, OTHERWISE USE THE ONE ABOVE ###########################################################
@click.command()
@click.argument('split_no', type=int)
@click.option('--experiment_name', type=str, default="")
def run_generator_with_resource_profiler(split_no: int, experiment_name: str = None):

    config = yaml.safe_load(open("config.yaml"))
    generator_name = config.get("generator_name")

    prof = ResourceProfiler()

    GeneratorClass = get_generator_class(generator_name)
    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config, split_no=split_no)

    metrics = {
        "model":            generator_name,
        "split_no":         split_no,
        "experiment_name":  experiment_name,
        "dataset":          config.get("dataset_name"),

        # Hardware — GPU or CPU
        "hardware_type":    "GPU" if torch.cuda.is_available() else "CPU",
        **(prof.get_gpu_info() if torch.cuda.is_available() else prof.get_cpu_info()),

        # Timing
        "train_time_sec":       None,
        "generation_time_sec":  None,
        "n_generated_samples":  None,
        "n_training_samples":   len(generator.X_train) if hasattr(generator, "X_train") else None,

        # Memory
        "train_peak_memory_gb": None,
        "gen_peak_memory_gb":   None,
        "memory_type":          "GPU" if torch.cuda.is_available() else "RAM",

        # Model
        "n_parameters_M":   None,
    }
    # ----------------------------------------------------
    # Skip special generator types if needed
    # ----------------------------------------------------
    if isinstance(generator, MultivariateDataGenerator):
        if config.get("generate", True):
            prof.start()
            syn_data, syn_lbl = generator.generate()
            gen_stats = prof.stop()
            metrics["generation_time_sec"] = gen_stats["elapsed_sec"]
            metrics["gen_peak_memory_gb"] = gen_stats["peak_memory_gb"]
            metrics["n_generated_samples"] = len(syn_data)
            generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)
        prof.save_metrics(metrics, split_no, experiment_name)
        return
    # ----------------------------------------------------
    # TRAINING PHASE
    # ----------------------------------------------------
    if not config.get("load_from_checkpoint", False):

        if config.get("train", False):

            prof.start()
            generator.train()
            train_stats = prof.stop()

            metrics["train_time_sec"] = train_stats["elapsed_sec"]
            metrics["train_peak_memory_gb"] = train_stats["peak_memory_gb"]

        else:
            generator.load_from_checkpoint()

    # ----------------------------------------------------
    # PARAMETER COUNT
    # ----------------------------------------------------
    mods = _get_module_for_params(generator)
    if mods:
        total = sum(
            sum(p.numel() for p in m.parameters() if p.requires_grad)
            for m in mods
        )
        metrics["n_parameters_M"] = round(total / 1e6, 3)

    # ----------------------------------------------------
    # GENERATION PHASE
    # ----------------------------------------------------
    if config.get("generate", True):

        prof.start()
        syn_data, syn_lbl = generator.generate()
        gen_stats = prof.stop()

        metrics["generation_time_sec"] = gen_stats["elapsed_sec"]
        metrics["gen_peak_memory_gb"] = gen_stats["peak_memory_gb"]
        metrics["n_generated_samples"] = len(syn_data)

        generator.save_synthetic_data(syn_data, syn_lbl, experiment_name)

    # ----------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------
    prof.save_metrics(metrics, split_no, generator_name)


#######################################################################################################################################################################

## your synthetic data will be saved accordingly to config.yaml
## e.g. data_splits/{dataset_name}/synthetic/{generator_name}/{experiment_name}
## change the corresponding keys in the config.yaml
@click.command()
@click.argument('split_no', type=int)
@click.argument('subtype', type=str)
@click.argument('num_samples', type=int)
@click.option('--experiment_name', type=str, default="")
def run_pretrained_generator_for_type(split_no: int, subtype:str, num_samples:int, experiment_name: str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    generator_name = config.get('generator_name')
    GeneratorClass = get_generator_class(generator_name)

    if not GeneratorClass:
        raise ValueError(f"Unknown generator name: {generator_name}")

    generator = GeneratorClass(config, split_no=split_no)

    if not isinstance(generator, MultivariateDataGenerator):
        generator.load_from_checkpoint()


    syn_data, syn_lbl = generator.generate_for_type(subtype, num_samples)
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
cli.add_command(run_pretrained_generator_for_type)
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