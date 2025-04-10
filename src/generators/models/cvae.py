import os
import click
import yaml
import torch
import numpy as np
import random
from typing import Dict, Any
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import pandas as pd

import torch.nn as nn
import numpy as np
import click

from generators.utils.ops import one_hot_embedding
from generators.utils.dataset import LoadDataset
from generators.utils.misc import mkdir
from generators.models.base import BaseDataGenerator

# force early initialization of cuda runtime.
# the script sometimes get abruptly killed without this
# it may be driver specific and as such, this case may not apply to you.
if torch.cuda.is_available:
    print(torch.cuda.is_available)
    torch.cuda.current_device()


@click.group()
def cli():
    pass


class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, beta=1, transform="none"):
        super(CVAE, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.beta = beta
        self.transform = transform

        self.fc_feat_x = nn.Sequential(
            nn.Linear(x_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_feat_y = nn.Sequential(nn.Linear(y_dim, 256), nn.ReLU())
        self.fc_feat_all = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        self.dec_z = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU())
        self.dec_y = nn.Sequential(nn.Linear(y_dim, 256), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, x_dim),
        )

        self.rec_crit = nn.MSELoss()
        self.encoder = [
            self.fc_feat_x,
            self.fc_feat_y,
            self.fc_feat_all,
            self.fc_logvar,
            self.fc_mu,
        ]
        self.decoder = [self.dec_z, self.dec_y, self.dec]
        self.encoder_params = list()
        for layer in self.encoder:
            self.encoder_params = self.encoder_params + list(layer.parameters())
        self.decoder_params = list()
        for layer in self.decoder:
            self.decoder_params = self.decoder_params + list(layer.parameters())

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        out = self.decode(self.reparameterize(mu, logvar), y)

        return mu, logvar, out

    def compute_loss(self, x, y, verbose=True):
        mu, logvar, rec = self.forward(x, y)

        rec_loss = self.rec_crit(rec, x)

        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )

        loss = rec_loss + self.beta * kl_loss

        if verbose:
            return {
                "loss": loss,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
                "mu": mu,
                "logvar": logvar,
                "rec": rec,
            }
        else:
            return {"loss": loss, "rec_loss": rec_loss, "kl_loss": kl_loss}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x, y):
        feat_x = self.fc_feat_x(x)
        feat_y = self.fc_feat_y(y)
        feat = torch.cat([feat_x, feat_y], dim=1)
        feat = self.fc_feat_all(feat)

        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)

        return mu, logvar

    def decode(self, z, y):
        dec_y = self.dec_y(y)
        dec_z = self.dec_z(z)
        out = self.dec(torch.cat([dec_z, dec_y], dim=1))

        if self.transform == "exp":
            out = out.exp()
        elif self.transform == "tahn":
            out = torch.tahn(out)
        elif self.transform == "sigmoid":
            out = torch.sigmoid(out)
        elif self.transform == "relu":
            out = torch.nn.ReLU()(out)
        return out

    def sample(self, n, data_loader, uniform_y=False, device="cpu"):
        """Returns (fake_data, fake_label) samples."""

        self.eval()

        fake_data = []
        fake_label = []
        total = 0

        while total <= n:
            for data_x, data_y in data_loader:
                if total > n:
                    break
                else:
                    z = torch.randn([data_x.shape[0], self.z_dim]).to(device)
                    if uniform_y:
                        data_y = torch.randint(0, self.y_dim, [data_x.shape[0]])
                    y = one_hot_embedding(data_y, num_classes=self.y_dim, device=device)
                fake_data.append(self.decode(z, y).detach().cpu().numpy())
                fake_label.append(data_y.cpu().numpy())

                total += len(data_x)

        fake_data = np.concatenate(fake_data)[:n]
        fake_label = np.concatenate(fake_label)[:n]

        return fake_data, fake_label
    
    def generate_for_subtype(self, subtype_label: int, num_samples: int = 100, device="cpu"):
        self.eval()

        # One-hot encode the target label
        y_onehot = one_hot_embedding(
            torch.tensor([subtype_label] * num_samples), 
            num_classes=self.y_dim,
            device=device
        )

        z = torch.randn([num_samples, self.z_dim]).to(device)
        fake_data = self.decode(z, y_onehot).detach().cpu().numpy()
        fake_label = np.array([subtype_label] * num_samples)

        return fake_data, fake_label




class CVAEDataGenerationPipeline(BaseDataGenerator):
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
        self.z_dim = self.args["model_config"]["z_dim"]
        self.beta = self.args["model_config"]["beta"]
        self.lr = self.args["model_config"]["lr"]
        self.transform = self.args["model_config"]["transform"]
        self.num_iters = self.args["train_config"]["num_iters"]
        self.batch_size = self.args["train_config"]["batch_size"]
        self.if_uniform_y = self.args["generation_config"]["uniform_y"]
        self.n_synth_samples = self.args["generation_config"]["n_synth_samples"]

        if self.generator_name == "cvae":
            self.experiment_name = f'pp_{self.preprocess}-bs_{self.batch_size}-iters_{self.num_iters}-beta_{self.args["model_config"]["beta"]}'
            self.enable_privacy = False
        elif self.generator_name == "dpcvae":
            self.experiment_name = f'pp_{self.preprocess}-bs_{self.batch_size}-iters_{self.num_iters}-beta_{self.args["model_config"]["beta"]}-eps_{self.args["privacy_config"]["target_epsilon"]}-clip_{self.args["privacy_config"]["max_norm"]}'
            self.enable_privacy = True
        else:
            raise NotImplementedError

        if self.enable_privacy:
            self.target_epsilon = self.args["privacy_config"]["target_epsilon"]
            self.target_delta = self.args["privacy_config"]["target_delta"]
            self.max_norm = self.args["privacy_config"]["max_norm"]

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
        self.setup_model_and_optimizer()

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

        self.dset = LoadDataset(
            data_X=X_train, data_y=y_train, preprocess=self.preprocess
        )

    def setup_model_and_optimizer(self):
        x_dim, y_dim = self.dset.get_dim()

        self.model = CVAE(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=self.z_dim,
            beta=self.beta,
            transform=self.transform,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        x_dim, y_dim = self.dset.get_dim()
        print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

        data_loader = DataLoader(self.dset, batch_size=self.batch_size, shuffle=True)

        print("Training started...")
        num_epochs = (self.num_iters * self.batch_size) // len(self.dset)
        print(f"Epochs: {num_epochs}")

        if self.enable_privacy:
            privacy_engine = PrivacyEngine()
            (
                self.model,
                self.optimizer,
                data_loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=data_loader,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_norm,
                epochs=num_epochs,
            )
        iters = 0
        for ep in range(num_epochs):
            for data_x, data_y in data_loader:
                iters += 1
                data_x = data_x.to(self.device)
                data_y = one_hot_embedding(
                    data_y, num_classes=y_dim, device=self.device
                )

                self.model.train()
                self.model.zero_grad()

                if self.enable_privacy:
                    output = self.model._module.compute_loss(
                        data_x, data_y, verbose=True
                    )
                else:
                    output = self.model.compute_loss(data_x, data_y, verbose=True)

                loss = output["loss"]
                loss.backward()
                self.optimizer.step()

                if self.enable_privacy:
                    self.optimizer.zero_grad()

            print(f"Epoch: {ep}, Loss: {loss}")

            # save model
            save_dict = {
                "iter": iters,
                "epoch": ep,
                "model_state_dict": (
                    self.model._module.state_dict()
                    if self.enable_privacy
                    else self.model.state_dict()
                ),
                "opt_state_dict": self.optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(
                    self.model_save_dir,  f"checkpoint_split_{self.split_no}.pkl"
                ),
            )

        if self.enable_privacy:
            self.model = self.model._module

    def load_from_checkpoint(self):
        print("Load from checkpoint.")
        checkpoint = torch.load(
            os.path.join(self.model_save_dir, f"checkpoint_split_{self.split_no}.pkl")
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["opt_state_dict"])



    def generate(self):
        print("Generate data.")

        data_loader = DataLoader(self.dset, batch_size=self.batch_size, shuffle=False)

        if self.n_synth_samples == -1:
            n_synth_samples = len(self.dset)
        else:
            n_synth_samples = self.n_synth_samples

        syn_data, syn_labels = self.model.sample(
            n_synth_samples, data_loader, self.if_uniform_y, device=self.device
        )
        syn_data, syn_labels = self.dset.xy_to_dataframe(syn_data, syn_labels)
        return syn_data, syn_labels


    def generate_for_type(self, subtype_label: int, num_samples: int = 100):
        """
        Wrapper to generate data for a specific subtype using the trained model.
        """
        label_idx = self.dset.label_encoder.transform([subtype_label])[0]

        self.model.eval()
        syn_data, syn_labels = self.model.generate_for_subtype(
            subtype_label=label_idx,
            num_samples=num_samples,
            device=self.device
        )
        return self.dset.xy_to_dataframe(syn_data, syn_labels)
    


    def save_synthetic_data(
        self,
        synthetic_features: pd.DataFrame,
        synthetic_labels: pd.DataFrame,
        experiment_name: str = "",
    ):
        
        BaseDataGenerator.save_synthetic_data(
            self, synthetic_features, synthetic_labels, self.experiment_name
        )


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("split_no", type=int)
def run_pipeline(config_path: str, split_no: int):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["generator_name"] not in ["dpcvae", "cvae"]:
        config["generator_name"] = "dpcvae"

    pipeline = CVAEDataGenerationPipeline(config, split_no=split_no)

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
