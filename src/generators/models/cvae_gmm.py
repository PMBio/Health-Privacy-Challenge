import os
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from typing import Dict, Any
from tqdm import tqdm

from generators.models.base import BaseDataGenerator
#from utils.data import RNAseqDataset

def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


## UPDATE: Swithcing to one-hot encoded disease from embedded disease for fair comparison
## Performance improvement with GAN might come from that...


def one_hot_embedding(labels, num_classes, device='cpu'):
    """
    Args:
        labels: tensor of shape [batch_size] with integer labels
        num_classes: total number of classes    
    Returns:
        one_hot: tensor of shape [batch_size, num_classes]
    """
    y = torch.eye(num_classes).to(device)

    return y[labels]



class CVAEGMMDataGenerator(BaseDataGenerator):
    def __init__(self, config: Dict[str, Any],  split_no: int = 1):
        super().__init__(config, split_no)
    
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.experiment_name = experiment_name
        self.random_seed = self.generator_config["random_seed"]

        # Load CVAE-GMM specific configurations
        self.cvae_config = self.generator_config
        self.latent_dim = self.cvae_config['model_config'].get('latent_dim', 64)
        self.n_components = self.cvae_config['model_config'].get('n_components', 3)
        self.batch_size = self.cvae_config['train_config'].get('batch_size', 32)
        self.epochs = self.cvae_config['train_config'].get('epochs', 100)
        self.lr = self.cvae_config['model_config'].get('learning_rate', 0.001)
        self.beta = self.cvae_config['model_config'].get('beta', 1.0)  # KL weight
        self.preprocess = self.cvae_config.get('preprocess', "standard")  

        self.condition_type = self.cvae_config['model_config'].get('condition_type', 'onehot')
        self.disease_embed_dim = self.cvae_config.get('disease_embed_dim', 20)
    
        
        # Network architecture
        self.encoder_hidden_dims = [1024, 512, 256]
        self.decoder_hidden_dims = [256, 512, 1024]
        
        #self.disease_embed_dim = self.cvae_config.get('disease_embed_dim', 20)
        self.target_dir = os.path.join(
            self.config["dir_list"]["home"],
            self.config["dir_list"]["data_save_dir"],
            self.dataset_name,
            "real",
        )

        if self.generator_name == "cvae_gmm":
            self.experiment_name = f'pp_{self.preprocess}-bs_{self.batch_size}-beta_{self.beta}-type={self.condition_type}'
            self.enable_privacy = False
        elif self.generator_name == "dpcvae_gmm":
            self.target_epsilon = self.cvae_config["privacy_config"]["target_epsilon"]
            self.max_norm = self.cvae_config["privacy_config"]["max_norm"]
            self.target_delta = self.cvae_config["privacy_config"]["target_delta"]
            self.experiment_name = (
                f'pp_{self.preprocess}-bs_{self.batch_size}-iters_{self.num_iters}-beta_{self.beta}-type={self.condition_type}'
                f'-eps_{self.target_epsilon }-clip_{self.clip}'
            )
            self.enable_privacy = True

        # Initialize models as None (will be created during training)
        self.cvae = None
        self.gmm_models = {}  # Dictionary to store GMM for each subgroup
        self.dataset = None
        self.optimizer = None
        self.disease_encoder = None

        
        # Training history
        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []

    def _load_data(self):
        temp_subtype_col_name = self.subtype_col_name #"task_label"

        X_train_df = pd.read_csv(
            os.path.join(self.target_dir, f"X_train_real_split_{self.split_no}.csv")
        )
        X_train = X_train_df.values
        y_train = pd.read_csv(
            os.path.join(self.target_dir, f"y_train_real_split_{self.split_no}.csv"),
        )

        # Create group labels
        y_train["group_label"] = (
            y_train[temp_subtype_col_name].astype(str)
        )

        self.X_train_features = X_train_df.columns.values
    
        # Extract disease  information from labels
        disease_labels = y_train[temp_subtype_col_name].values

        
        return X_train, disease_labels
    
    def _create_models(self, n_genes, n_diseases):
        """Create CVAE model"""
        self.cvae = CVAE(
            n_genes=n_genes,
            latent_dim=self.latent_dim,
            n_diseases=n_diseases,
            encoder_hidden_dims=self.encoder_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            
            condition_type=self.condition_type,
            disease_embed_dim=self.disease_embed_dim
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.cvae.parameters(), lr=self.lr)
    
    def cvae_loss(self, recon_x, x, mu, logvar):
        """Calculate CVAE loss (reconstruction + KL divergence)"""
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean') # changed mean to sum
        
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
    
    
    def train_step(self, data, disease):
        """Single training step"""
        self.optimizer.zero_grad()
        
        recon_data, mu, logvar = self.cvae(data, disease)
        
        total_loss, recon_loss, kl_loss = self.cvae_loss(recon_data, data, mu, logvar)
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), recon_loss.item(), kl_loss.item()
    
    def train(self):
        """Train the CVAE model and fit GMM for each subgroup"""
        print(f"Training CVAE-GMM on device: {self.device}")
        
        # Load data
        expression_data, disease_labels = self._load_data()
        
        # Create dataset
        self.dataset = RNAseqDataset(expression_data, disease_labels, self.preprocess)
        self.disease_encoder = self.dataset.disease_encoder


        logging.info(self.dataset.expression_data.shape)
        dataloader = DataLoader(self.dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                drop_last=False
                                )
        
        # Create CVAE model
        n_genes = expression_data.shape[1]
        self._create_models(n_genes, self.dataset.n_diseases)
        
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Number of genes: {n_genes}")
        print(f"Number of diseases: {self.dataset.n_diseases}")
        
        # Training loop for CVAE
        self.cvae.train()
        
        for epoch in tqdm(range(self.epochs), desc="Training CVAE"):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            n_batches = 0
            
            for batch in dataloader:
                data = batch['expression'].to(self.device)
                disease = batch['disease'].to(self.device)
                
                total_loss, recon_loss, kl_loss = self.train_step(data, disease)
                
                epoch_total_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_kl_loss += kl_loss
                n_batches += 1
            
            # Average losses for epoch
            avg_total_loss = epoch_total_loss / n_batches
            avg_recon_loss = epoch_recon_loss / n_batches
            avg_kl_loss = epoch_kl_loss / n_batches
            
            self.train_losses.append(avg_total_loss)
            self.recon_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] - Total: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
        
        # After CVAE training, fit GMM for each subgroup
        print("\nFitting GMM models for each subgroup...")
        self._fit_gmm_models()
        
        # Save model checkpoint
        self._save_checkpoint()
        
        print("Training completed!")
    
    def _fit_gmm_models(self, min_samples=10):
        """Fit GMM to latent representations for each disease subgroup"""
        self.cvae.eval()
        
        labels_df = pd.DataFrame({
            'disease': self.dataset.disease_encoder.inverse_transform(self.dataset.encoded_diseases)
        })

        print(labels_df)
        print(self.dataset.encoded_diseases)
        
        # Get all latent representations
        all_latents = []
        all_diseases = []

        
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                batch = self.dataset[idx]
                data = batch['expression'].unsqueeze(0).to(self.device)
                disease = batch['disease'].unsqueeze(0).to(self.device)

                
                ## fitting on repametrized 
                mu, logvar = self.cvae.encode(data, disease)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z =  mu + eps * std   
                all_latents.append(z.cpu().numpy())

                all_diseases.append(self.dataset.encoded_diseases[idx])
        
        all_latents = np.vstack(all_latents)
        
        # Fit GMM for each subgroup
        unique_combinations = labels_df.drop_duplicates().values
        
        for disease_name in unique_combinations:
            print(disease_name)
            disease_idx = list(self.disease_encoder.classes_).index(disease_name)
            print(disease_idx)
            # Get latents for this subgroup
            mask = (np.array(all_diseases) == disease_idx)
            subgroup_latents = all_latents[mask]
            print(subgroup_latents.shape)

            if len(subgroup_latents) < min_samples:
                print(f"Warning: Subgroup {disease_name} has only {len(subgroup_latents)} samples, skipping...")
                continue

            
            # Fit GMM
            gmm = GaussianMixture(n_components=self.n_components, 
                                  covariance_type='full', 
                                  random_state=self.random_seed)
            gmm.fit(subgroup_latents)
            
            self.gmm_models[disease_name[0]] = gmm
            print(f"Fitted GMM for {disease_name} with {self.n_components} components ({len(subgroup_latents)} samples)")

    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(
            self.config["dir_list"]["home"],
            self.config["dir_list"]["data_save_dir"], 
            self.dataset_name,
            "synthetic",
            "checkpoint",
            self.generator_name,
            self.experiment_name
        )
        check_dirs(checkpoint_dir)

        
        checkpoint = {
            'cvae_state_dict': self.cvae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dataset_scaler': self.dataset.scaler,
            'disease_encoder': self.dataset.disease_encoder,
            'gmm_models': self.gmm_models,
            'train_losses': self.train_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses,
            'config': self.config,
            'n_genes': self.cvae.n_genes,
            'n_diseases': self.dataset.n_diseases,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"cvae_gmm_checkpoint_{self.split_no}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Model checkpoint saved to: {checkpoint_path}")
    
    def load_from_checkpoint(self, checkpoint_path=None):
        """Load model from checkpoint"""
        if checkpoint_path is None:
            checkpoint_dir = os.path.join(
                self.config["dir_list"]["home"],
                self.config["dir_list"].get("model_save_dir", "models"),
                self.dataset_name,
                self.generator_name,
                self.experiment_name
            )
            checkpoint_path = os.path.join(checkpoint_dir, f"cvae_gmm_checkpoint_{self.split_no}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate models with saved dimensions
        n_genes = checkpoint['n_genes']
        n_diseases = checkpoint['n_diseases']

        
        self._create_models(n_genes, n_diseases)
        
        # Load model states
        self.cvae.load_state_dict(checkpoint['cvae_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load dataset encoders and scaler
        self.dataset_scaler = checkpoint['dataset_scaler']
        self.disease_encoder = checkpoint['disease_encoder']
        
        # Load GMM models
        self.gmm_models = checkpoint['gmm_models']
        
        # Load training history
        self.train_losses = checkpoint['train_losses']
        self.recon_losses = checkpoint['recon_losses']
        self.kl_losses = checkpoint['kl_losses']
        
        print(f"Model loaded from: {checkpoint_path}")

    def generate(self, min_samples=10):
        """Generate synthetic data with volume-matching, skip subgroups with < min_samples"""
        if self.cvae is None:
            raise ValueError("Model not trained or loaded. Call train() or load_from_checkpoint() first.")
        
        self.cvae.eval()
        
        synthetic_data_list = []
        synthetic_labels_list = []
        
        # Compute counts per disease subgroup
        labels_df = pd.DataFrame({
            self.subtype_col_name: self.dataset.disease_encoder.inverse_transform(self.dataset.encoded_diseases)
        })
        
        # Group by disease and get counts
        group_counts = labels_df.groupby([self.subtype_col_name]).size()
        
        with torch.no_grad():
            for disease_name, count in group_counts.items():
                if count < min_samples:
                    print(f"Skipping subgroup {disease_name} (n={count} < {min_samples})")
                    continue  # skip small groups
                
                # Get GMM for this subgroup
                if (disease_name) not in self.gmm_models:
                    print(f"Warning: No GMM found for {disease_name}")
                    continue
                
                gmm = self.gmm_models[(disease_name)]
                
                # Sample from GMM
                latent_samples, _ = gmm.sample(count)
                latent_tensor = torch.FloatTensor(latent_samples).to(self.device)
                
                disease_idx = list(self.disease_encoder.classes_).index(disease_name)
                disease_tensor = torch.LongTensor([disease_idx] * count).unsqueeze(1).to(self.device)

                
                # Decode samples
                synthetic_batch = self.cvae.decode(latent_tensor, disease_tensor)
                     # CRITICAL FIX: Inverse transform to original scale
                synthetic_batch_scaled = self.dataset.scaler.inverse_transform(
                    synthetic_batch.cpu().numpy()
                )
                synthetic_data_list.append(synthetic_batch_scaled)
                #synthetic_data_list.append(synthetic_batch.cpu().numpy())
                
                batch_labels = pd.DataFrame({
                    self.subtype_col_name: [disease_name] * count
                })
                synthetic_labels_list.append(batch_labels)
        
        # Combine all synthetic data
        if len(synthetic_data_list) == 0:
            raise ValueError("No subgroups meet the minimum sample requirement.")
        
        synthetic_features = np.vstack(synthetic_data_list)
        synthetic_labels = pd.concat(synthetic_labels_list, ignore_index=True)
        
        sample_names = [f"{i}" for i in range(synthetic_features.shape[0])]
        
        synthetic_features_df = pd.DataFrame(
            synthetic_features,
            index=sample_names,
            columns=self.X_train_features
        )
        synthetic_labels.index = sample_names

        # Shuffling before saving
        shuffled_indices = np.random.permutation(synthetic_features_df.shape[0])
        synthetic_features_df = synthetic_features_df.iloc[shuffled_indices].reset_index(drop=True)
        synthetic_labels = synthetic_labels.iloc[shuffled_indices].reset_index(drop=True)

        # Save synthetic data
        self.save_synthetic_data(synthetic_features_df, synthetic_labels, self.experiment_name)
        
        return synthetic_features_df, synthetic_labels

    def generate_for_type(self, disease_type=None, n_samples=100):
       pass




class CVAE(nn.Module):
    """Conditional Variational Autoencoder conditioned on disease"""
    
    def __init__(self, n_genes=1000, latent_dim=64, n_diseases=5,
                 encoder_hidden_dims=[1024, 512, 256],
                 decoder_hidden_dims=[256, 512, 1024],
                 condition_type="onehot", disease_embed_dim=20):
        super(CVAE, self).__init__()
        
        self.n_genes = n_genes
        self.n_diseases = n_diseases
        self.latent_dim = latent_dim
        self.condition_type = condition_type
        
        # Embedding layers for conditions
        #self.disease_embedding = nn.Embedding(n_diseases, disease_embed_dim)
        # condition_dim = n_diseases 

        if condition_type == "embedding":
            self.disease_embedding = nn.Embedding(n_diseases, disease_embed_dim)
            #nn.init.xavier_uniform_(self.disease_embedding.weight) ## wasn't helpful
            condition_dim =  disease_embed_dim
        else:
            condition_dim = n_diseases
        
        # Encoder
        encoder_layers = []
        input_dim = n_genes + condition_dim
        prev_dim = input_dim
        
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim + condition_dim
        prev_dim = input_dim
        
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, n_genes))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def get_condition_embedding(self, disease):
        """Convert disease indices to one-hot encoding"""
        # Ensure shape [batch_size]
        disease = disease.view(-1)
        # Create one-hot encoding
        #disease_onehot = one_hot_embedding(disease, 
        #                                   num_classes=self.n_diseases, 
        #                                   device=disease.device)
        
        if self.condition_type == "embedding":
            return self.disease_embedding(disease)
        else:
            return one_hot_embedding(disease, num_classes=self.n_diseases, device=disease.device)



    def encode(self, x, disease):
        """Encode input to latent space"""
        condition = self.get_condition_embedding(disease)
        x_cond = torch.cat([x, condition], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, disease):
        """Decode from latent space"""
        condition = self.get_condition_embedding(disease)
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)
    
    def forward(self, x, disease):
        """Forward pass"""
        mu, logvar = self.encode(x, disease)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, disease)
        return recon_x, mu, logvar


class RNAseqDataset(Dataset):
    """Dataset class for RNAseq data with disease conditioning"""
    
    def __init__(self, expression_data, disease_labels, scaler=None):
        self.expression_data = expression_data
        self.disease_labels = disease_labels
        
        # Encode disease labels
        self.disease_encoder = LabelEncoder()
        self.encoded_diseases = self.disease_encoder.fit_transform(disease_labels)
        self.n_diseases = len(self.disease_encoder.classes_)
        

        logging.info(f"Disease types found: {self.disease_encoder.classes_}")
        
        # Normalize expression data
        if scaler == "standard" or scaler is None:
            self.scaler = StandardScaler()
            self.normalized_data = self.scaler.fit_transform(expression_data)
            logging.info("Used StandardScaler()..")
        else:
            self.normalized_data = expression_data


    def __len__(self):
        return len(self.expression_data)
    
    def __getitem__(self, idx):
        return {
            'expression': torch.FloatTensor(self.normalized_data[idx]),
            'disease': torch.LongTensor([self.encoded_diseases[idx]])
        }


def create_example_config():
    """Create example configuration for testing"""
    config = {
        "generator_name": "cvae_gmm",
        "cvae_gmm_config": {
            "latent_dim": 64,
            "n_components": 3,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "beta": 1.0,
            "disease_embed_dim": 20,
        }
    }
    return config