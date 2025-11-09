import click
import yaml
import os
import sys
import importlib

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

# mapping model names to their respective classes
# be consistent with config.yaml about the use of names 
mia_classes = {
    'domias_baselines': ('models.baseline', 'DOMIASBaselineModels'),
    'sc_domias_baselines': ('models.sc_baseline', 'DOMIASSingleCellBaselineModels')
}

## dynamic import to avoid package versioning errors 
def get_mia_class(mia_name):
    if mia_name in mia_classes:
        module_name, class_name = mia_classes[mia_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Unknown generator name: {mia_name}")



@click.group()
def cli():
    pass



## the outputs of the attack are always saved under
## /{home_dir}/results/mia/{dataset_name}/{attacker_name}/{generator_model}/{experiment_name}
@click.command()
@click.argument('synthetic_file', type=click.Path(exists=True))
@click.argument('mmb_test_file', type=click.Path(exists=True))
@click.argument('mia_experiment_name', type=str, default="")
@click.option('--mmb_labels_file', type=click.Path(exists=True), default=None)
@click.option('--test_on_real', type=bool, default=False)
@click.option('--reference_file', type=click.Path(exists=True), default=None)
def run_mia(synthetic_file:str, 
            mmb_test_file:str, 
            mia_experiment_name:str = "",
            mmb_labels_file:str = None,
            test_on_real: bool = False, 
            reference_file:str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    attack_model = config.get('attack_model')
    MIAClass = get_mia_class(attack_model)

    if not MIAClass:
        raise ValueError(f"Unknown MIA model name: {attack_model}")

    mia_model = MIAClass(config, 
                         synthetic_file,
                         mmb_test_file,
                         mmb_labels_file,
                         mia_experiment_name,
                         reference_file,
                         test_on_real)
    
    predictions, y_test = mia_model.run_attack()
    mia_model.save_predictions(predictions)

    if y_test is not None:
        mia_model.evaluate_attack(predictions, 
                                 y_test, 
                                "evaluation_results.csv")
    


## the outputs of the attack are always saved under
## /{home_dir}/results/mia/{dataset_name}/{attacker_name}/{generator_model}/{experiment_name}
@click.command()
@click.argument('synthetic_file', type=click.Path(exists=True))
@click.argument('mmb_test_file', type=click.Path(exists=True))
@click.argument('mia_experiment_name', type=str, default="")
@click.option('--mmb_labels_file', type=click.Path(exists=True), default=None)
@click.option('--reference_file', type=click.Path(exists=True), default=None)
def run_singlecell_mia(synthetic_file:str, 
                        mmb_test_file:str, 
                        mia_experiment_name:str = "",
                        mmb_labels_file:str = None,
                        reference_file:str = None):
    # Load the config file
    configfile = "config.yaml"
    config = yaml.safe_load(open(configfile))

    attack_model = config.get('attack_model')
    MIAClass = get_mia_class(attack_model)

    if not MIAClass:
        raise ValueError(f"Unknown MIA model name: {attack_model}")

    mia_model = MIAClass(config, 
                         synthetic_file,
                         mmb_test_file,
                         mmb_labels_file,
                         mia_experiment_name,
                         reference_file)
    
    predictions, y_test = mia_model.run_attack()
    mia_model.save_predictions(predictions)

    if y_test is not None:
        #donor level averaging 
        grp_preds, grp_y = mia_model.perform_donor_level_avg(predictions, y_test)
        mia_model.evaluate_attack(grp_preds, 
                                  grp_y, 
                                  "evaluation_results.csv")





cli.add_command(run_singlecell_mia)
cli.add_command(run_mia)
if __name__ == '__main__':
    cli()


