import os
import click
import random
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any
import torch

from snsynth.pytorch.nn import DPCTGAN
from snsynth.pytorch import PytorchDPSynthesizer

from generators.models.base import BaseDataGenerator
from generators.utils.misc import mkdir

# force early initialization of cuda runtime.
# the script sometimes get abruptly killed without this
# it may be driver specific and as such, this case may not apply to you.
if torch.cuda.is_available:
    torch.cuda.current_device()


@click.group()
def cli():
    pass


class DPCTGANDataGenerationPipeline(BaseDataGenerator):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        self.config = config
        self.split_no = split_no

        self.data_path = os.path.join(
            self.config["dir_list"]["home"], self.config["dir_list"]["data_save_dir"]
        )
        self.dataset_name = self.config["dataset_config"]["name"]
        self.generator_name = self.config["generator_name"]

        self.args = self.config[self.generator_name + "_config"]
        self.device = torch.device(self.args["device_type"])
        self.random_seed = self.args["random_seed"]
        self.preprocess = self.args["preprocess"]
        self.batch_size = self.args["train_config"]["batch_size"]
        self.num_iters = self.args["train_config"]["num_iters"]
        self.n_synth_samples = self.args["generation_config"]["n_synth_samples"]

        self.target_epsilon = self.args["privacy_config"]["target_epsilon"]
        self.preprocessor_eps_multiplier = self.args["privacy_config"][
            "preprocessor_eps_multiplier"
        ]
        self.target_delta = self.args["privacy_config"]["target_delta"]
        self.max_norm = self.args["privacy_config"]["max_norm"]

        self.experiment_name = f"pp_{self.preprocess}-bs_{self.batch_size}-iters_{self.num_iters}-eps_{self.target_epsilon}-clip_{self.max_norm}"

        self.model_save_dir = os.path.join(
            self.data_path,
            self.dataset_name,
            "synthetic",
            "checkpoint",
            self.generator_name,
            self.experiment_name,
        )

        self.real_save_dir = os.path.join(self.data_path, self.dataset_name, "real")
        mkdir(self.model_save_dir)

        self.initialize_random_seeds()
        self.load_dataset()
        self.setup_model()

    def initialize_random_seeds(self):
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        if self.device.type == "cuda":
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

    def load_dataset(self):
        X_train = pd.read_csv(
            os.path.join(self.real_save_dir, f"X_train_real_split_{self.split_no}.csv")
        ).astype(float)
        y_train = pd.read_csv(
            os.path.join(self.real_save_dir, f"y_train_real_split_{self.split_no}.csv")
        )
        self.label_name = y_train.columns.to_list()
        self.dset = pd.concat([X_train, y_train], axis=1)

    def setup_model(self):
        num_epochs = (self.num_iters * self.batch_size) // len(self.dset)
        print(f"Epochs: {num_epochs}")
        self.model = PytorchDPSynthesizer(
            epsilon=self.target_epsilon,
            gan=DPCTGAN(
                epochs=num_epochs,
                epsilon=self.target_epsilon,
                delta=self.target_delta,
                max_per_sample_grad_norm=self.max_norm,
                cuda=True if self.device.type == "cuda" else False,
                batch_size=self.batch_size,
                verbose=True,
            ),
            preprocessor=None,
        )

    def train(self):
        print("Training started...")
        train_data = self.dset

        discrete_columns = train_data.select_dtypes(include=["object"]).columns

        self.model.fit(
            train_data,
            categorical_columns=discrete_columns,
            continuous_columns=list(set(train_data.columns) - set(discrete_columns)),
            # ordinal_columns=list(set(train_data.columns) - set(discrete_columns)),
            preprocessor_eps=self.preprocessor_eps_multiplier
            * len(list(set(train_data.columns) - set(discrete_columns))),
            nullable=True,
        )
        print(self.model.epsilon)
        # save model
        torch.save(
            self.model,
            os.path.join(self.model_save_dir, f"checkpoint_split_{self.split_no}.pkl"),
        )

    def load_from_checkpoint(self):
        print("Load from checkpoint.")
        self.model = torch.load(
            os.path.join(self.model_save_dir, f"checkpoint_split_{self.split_no}.pkl")
        )

    def generate(self):
        print("Generate data.")
        if self.n_synth_samples == -1:
            n_synth_samples = len(self.dset)
        else:
            n_synth_samples = self.n_synth_samples

        syn_data = self.model.sample(n_synth_samples)
        syn_labels = syn_data[self.label_name]
        syn_data = syn_data.drop(columns=self.label_name)
        return syn_data, syn_labels

    def save_synthetic_data(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.DataFrame,
        experiment_name: str = "",
    ):
        BaseDataGenerator.save_synthetic_data(
            self, synthetic_features, synthetic_labels, self.experiment_name
        )

    def generate_for_type(self):
        pass


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("split_no", type=int)
def run_pipeline(config_path: str, split_no: int):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["generator_name"] != "dpctgan":
        config["generator_name"] = "dpctgan"

    pipeline = DPCTGANDataGenerationPipeline(config, split_no=split_no)

    if not config.get("load_from_checkpoint", False):
        if config.get("train", False):
            pipeline.train()
    else:
        pipeline.load_from_checkpoint()

    if config.get("generate", False):
        syn_data, syn_labels = pipeline.generate()
        pipeline.save_synthetic_data(syn_data, syn_labels)


# cli.add_command(run_pipeline)
# if __name__ == "__main__":
#     # force early initialization of cuda runtime.
#     # the script sometimes get abruptly killed without this
#     # it may be driver specific and as such, this case may not apply to you.
#     if torch.cuda.is_available:
#         torch.cuda.current_device()

#     # Run CLI
#     cli()
