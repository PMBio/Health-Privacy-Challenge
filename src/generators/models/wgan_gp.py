import os
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from generators.models.base import BaseDataGenerator

def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

## WGAN-GP Gulrajani et al., 2017,

class WGANGPDataGenerator(BaseDataGenerator):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        super().__init__(config, split_no)
    
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = self.generator_config["random_seed"]

        # Load WGAN-GP specific configurations
        self.wgan_config = self.generator_config
        self.noise_dim = self.wgan_config['model_config'].get('latent_dim', 64)
        self.n_critic = self.wgan_config['model_config'].get('n_critic', 5)
        self.lambda_gp = self.wgan_config['model_config'].get('lambda_gp', 10)
        self.batch_size = self.wgan_config['train_config'].get('batch_size', 32)
        self.epochs = self.wgan_config['train_config'].get('epochs', 100)
        self.lr = self.wgan_config['model_config'].get('learning_rate', 0.0001)
        self.preprocess = self.wgan_config.get('preprocess', "standard")  
        #self.hidden_dims_g = self.wgan_config.get('generator_hidden_dims', [512, 256, 128])
        #self.hidden_dims_d = self.wgan_config.get('discriminator_hidden_dims', [128, 256, 512])
        self.hidden_dims_g = [1024, 512, 256] #[512, 256, 128]
        self.hidden_dims_d = [256, 512] #[128, 256, 512]
        self.condition_type = self.wgan_config['model_config'].get('condition_type', 'onehot')
        self.disease_embed_dim = self.wgan_config.get('disease_embed_dim', 20)

        self.target_dir = os.path.join(
            self.config["dir_list"]["home"],
            self.config["dir_list"]["data_save_dir"],
            self.dataset_name,
            "real",
        )

        if self.generator_name == "wgan_gp":
            self.experiment_name = f'pp_{self.preprocess}-bs_{self.batch_size}-nc_{self.n_critic}-nd_{self.noise_dim}-lgp_{self.lambda_gp}-type={self.condition_type}'
            self.enable_privacy = False
        elif self.generator_name == "dpwgan_dp":
            self.target_epsilon = self.cvae_config["privacy_config"]["target_epsilon"]
            self.max_norm = self.cvae_config["privacy_config"]["max_norm"]
            self.target_delta = self.cvae_config["privacy_config"]["target_delta"]

            self.experiment_name = f'pp_{self.preprocess}-bs_{self.batch_size}-nc_{self.n_critic}-nd_{self.noise_dim}-lgp_{self.lambda_gp}-type={self.condition_type}-eps_{self.target_epsilon }-clip_{self.clip}'
            self.enable_privacy = True

        # Initialize models as None (will be created during training)
        self.generator = None
        self.discriminator = None
        self.dataset = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.disease_encoder = None

        
        # Training history
        self.d_losses = []
        self.g_losses = []


    #### update.....    
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
        """Create generator and discriminator models"""
        self.generator = Generator(
            noise_dim=self.noise_dim,
            n_genes=n_genes,
            n_diseases=n_diseases,
            hidden_dims=self.hidden_dims_g,

            condition_type=self.condition_type,
            disease_embed_dim=self.disease_embed_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            n_genes=n_genes,
            n_diseases=n_diseases,
            hidden_dims=self.hidden_dims_d,
            
            condition_type=self.condition_type,
            disease_embed_dim=self.disease_embed_dim
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.9))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.0, 0.9))


    def _create_onehot_condition(self, disease_indices, n_diseases):
        """Create one-hot encoded condition vectors from disease indices"""
        batch_size = len(disease_indices)
        onehot = torch.zeros(batch_size, n_diseases)
        onehot[torch.arange(batch_size), disease_indices] = 1.0

        return onehot.to(self.device)
    
    def gradient_penalty(self, real_data, fake_data, disease):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand(real_data.size())
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = interpolated.to(self.device)
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated, disease)
        
        gradients = autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(d_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_data, disease_indices):
        """Single training step"""
        batch_size = real_data.size(0)
        n_diseases = len(self.disease_encoder.classes_)

        if self.condition_type == "onehot":
            disease = self._create_onehot_condition(disease_indices, n_diseases)
        else:
            disease = disease_indices  # use as integer indices for embeddings
            ##TODO: is this enough though???? 
        

        # Train Discriminator
        d_loss_total = 0
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()
            
            # Real data
            d_real = self.discriminator(real_data, disease)
            
            # Fake data
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_data = self.generator(noise, disease)
            d_fake = self.discriminator(fake_data.detach(), disease)
            
            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data, disease)
            
            # Discriminator loss
            d_loss = d_fake.mean() - d_real.mean() + gp
            d_loss.backward()
            self.d_optimizer.step()
            
            d_loss_total += d_loss.item()

        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_data = self.generator(noise, disease)
        g_fake = self.discriminator(fake_data, disease)
        
        g_loss = -g_fake.mean()
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss_total / self.n_critic, g_loss.item()
    
    def train(self):
        """Train the WGAN-GP model"""
        print(f"Training WGAN-GP on device: {self.device}")
        # Load data
        expression_data, disease_labels = self._load_data()
        
        # Create dataset
        self.dataset = RNAseqDataset(expression_data, disease_labels, self.preprocess)
        self.disease_encoder = self.dataset.disease_encoder

        logging.info(self.dataset.expression_data.shape)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        # Create models
        n_genes = expression_data.shape[1]
        self._create_models(n_genes, self.dataset.n_diseases)
        
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Number of genes: {n_genes}")
        print(f"Number of diseases: {self.dataset.n_diseases}")
        
        # Training loop
        self.generator.train()
        self.discriminator.train()
        
        for epoch in tqdm(range(self.epochs), desc="Training WGAN-GP"):
            epoch_d_loss = 0
            epoch_g_loss = 0
            n_batches = 0
            
            for batch in dataloader:
                real_data = batch['expression'].to(self.device)
                disease = batch['disease'].to(self.device)
     
                
                d_loss, g_loss = self.train_step(real_data, disease)
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                n_batches += 1
            
            # Average losses for epoch
            avg_d_loss = epoch_d_loss / n_batches
            avg_g_loss = epoch_g_loss / n_batches
            
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
        
        # Save model checkpoint
        self._save_checkpoint()
        
        print("Training completed!")


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
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'dataset_scaler': self.dataset.scaler,
            'disease_encoder': self.dataset.disease_encoder,
            'd_losses': self.d_losses,
            'g_losses': self.g_losses,
            'config': self.config,
            'n_genes': self.generator.n_genes,
            'n_diseases': self.dataset.n_diseases,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"wgan_gp_checkpoint_{self.split_no}.pth")
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
            checkpoint_path = os.path.join(checkpoint_dir, f"wgan_gp_checkpoint_{self.split_no}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate models with saved dimensions
        n_genes = checkpoint['n_genes']
        n_diseases = checkpoint['n_diseases']
        
        self._create_models(n_genes, n_diseases)
        
        # Load model states
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # Load dataset encoders and scaler
        self.dataset_scaler = checkpoint['dataset_scaler']
        self.disease_encoder = checkpoint['disease_encoder']
        
        # Load training history
        self.d_losses = checkpoint['d_losses']
        self.g_losses = checkpoint['g_losses']
        
        print(f"Model loaded from: {checkpoint_path}")


    def generate(self, min_samples=10):
        """Generate synthetic data with volume-matching, skip subgroups with < min_samples"""
        if self.generator is None:
            raise ValueError("Model not trained or loaded. Call train() or load_from_checkpoint() first.")
        
        self.generator.eval()
        
        synthetic_data_list = []
        synthetic_labels_list = []
        
        # Compute counts per disease subgroup
        labels_df = pd.DataFrame({
            self.subtype_col_name: self.dataset.disease_encoder.inverse_transform(self.dataset.encoded_diseases),
        })
        
        # Group by disease and get counts
        group_counts = labels_df.groupby([self.subtype_col_name]).size()
        
        with torch.no_grad():
            for disease_name, count in group_counts.items():
                if count < min_samples:
                    print(f"Skipping subgroup {disease_name}  (n={count} < {min_samples})")
                    continue  # skip small groups
                
                disease_idx = list(self.disease_encoder.classes_).index(disease_name)
                noise = torch.randn(count, self.noise_dim).to(self.device)
                #disease_tensor = torch.LongTensor([disease_idx] * count).unsqueeze(1).to(self.device)
                disease_indices = torch.LongTensor([disease_idx] * count).to(self.device)

                if self.condition_type == "embedding":
                    disease_tensor = disease_indices.long()
                else:
                    disease_tensor = self._create_onehot_condition(disease_indices, len(self.disease_encoder.classes_))

                synthetic_batch = self.generator(noise, disease_tensor)
                # CRITICAL FIX: Inverse transform to original scale
                synthetic_batch_scaled = self.dataset.scaler.inverse_transform(
                    synthetic_batch.cpu().numpy()
                )
                synthetic_data_list.append(synthetic_batch_scaled)
                
                batch_labels = pd.DataFrame({
                    self.subtype_col_name: [disease_name] * count,
                })
                synthetic_labels_list.append(batch_labels)
        
        # Combine all synthetic data
        if len(synthetic_data_list) == 0:
            raise ValueError("No subgroups meet the minimum sample requirement.")
        
        synthetic_features = np.vstack(synthetic_data_list)
        synthetic_labels = pd.concat(synthetic_labels_list, ignore_index=True)
        
        #gene_names = [f"Gene_{i}" for i in range(synthetic_features.shape[1])]
        sample_names = [f"{i}" for i in range(synthetic_features.shape[0])]
        
        synthetic_features_df = pd.DataFrame(
            synthetic_features,
            index=sample_names,
            columns=self.X_train_features
        )
        synthetic_labels.index = sample_names

        ### shuffling before saving 
        shuffled_indices = np.random.permutation(synthetic_features_df.shape[0])
        synthetic_features_df = synthetic_features_df.iloc[shuffled_indices].reset_index(drop=True)
        synthetic_labels = synthetic_labels.iloc[shuffled_indices].reset_index(drop=True)

        # Save synthetic data
        self.save_synthetic_data(synthetic_features_df, synthetic_labels, self.experiment_name)
        
        return synthetic_features_df, synthetic_labels


    
    def generate_for_type(self, disease_type=None, n_samples=100):
        pass



# Example configuration for testing
def create_example_config():
    """Create example configuration for testing"""
    config = {
        "generator_name": "wgan_gp",
        "dataset": {
            "name": "example_rnaseq",
            "data_dir": "./data",
            "subtype_col_name": "disease",
        },
        "wgan_gp_config": {
            "dir_feature_name": "lambda_gp",
            "lambda_gp": 10,
            "noise_dim": 100,
            "n_critic": 5,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.0001,
            "generator_hidden_dims": [512, 256, 128],
            "discriminator_hidden_dims": [128, 256, 512],
            "disease_embed_dim": 50,
            "n_synthetic_samples": 1000
        },
        "save": {
            "home_dir": "./results",
            "data_save_dir": "synthetic_data",
            "model_save_dir": "models"
        }
    }
    return config



class Generator(nn.Module):
    """Generator network conditioned on disease"""
    
    def __init__(self, noise_dim=100, n_genes=1000, n_diseases=5, 
                 hidden_dims=[512, 256, 128], condition_type="onehot", disease_embed_dim=20):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.n_genes = n_genes
        self.condition_type = condition_type
        
        if condition_type == "embedding":
            self.disease_embedding = nn.Embedding(n_diseases, disease_embed_dim)
            #nn.init.xavier_uniform_(self.disease_embedding.weight) ## wasn't helpful
            input_dim = noise_dim + disease_embed_dim
        else:
            input_dim = noise_dim + n_diseases
        
        # Embedding layers for conditions
        #self.disease_embedding = nn.Embedding(n_diseases, disease_embed_dim)

        # Calculate input dimension
        #input_dim = noise_dim + disease_embed_dim
        
        # Build generator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.extend([
            nn.Linear(prev_dim, n_genes)
            #nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, noise, condition):
        """
        Args:
            noise: [batch_size, noise_dim]
            condition: [batch_size, cond_dim] - one-hot encoded
        """
        if self.condition_type == "embedding":
            cond_vec = self.disease_embedding(condition)
        else:
            cond_vec = condition
        x = torch.cat([noise, cond_vec], dim=1)

        return self.model(x)
    

class Discriminator(nn.Module):
    """Discriminator network conditioned on disease"""
    
    def __init__(self, n_genes=1000, n_diseases=5,  hidden_dims=[128, 256, 512],
                condition_type="onehot", disease_embed_dim=20):
        super(Discriminator, self).__init__()

        self.condition_type = condition_type
        if condition_type == "embedding":
            self.disease_embedding = nn.Embedding(n_diseases, disease_embed_dim)
            #nn.init.xavier_uniform_(self.disease_embedding.weight) ## wasn't helpful
            input_dim = n_genes + disease_embed_dim
        else:
            input_dim = n_genes + n_diseases
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
            
        # Output layer (no activation for WGAN)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        

    def forward(self, expression, disease):
        """
        Args:
            expression: [batch_size, n_genes]
            condition: [batch_size, cond_dim] - one-hot encoded
        """
        if self.condition_type == "embedding":
            cond_vec = self.disease_embedding(disease)
        else:
            cond_vec = disease
        x = torch.cat([expression, cond_vec], dim=1)
        return self.model(x)

    



class RNAseqDataset(Dataset):
    """Dataset class for RNAseq data with disease and age conditioning"""
    
    def __init__(self, expression_data, disease_labels, scaler=None):
        self.expression_data = expression_data
        self.disease_labels = disease_labels

        # Encode disease labels
        self.disease_encoder = LabelEncoder()
        self.encoded_diseases = self.disease_encoder.fit_transform(disease_labels)
        self.n_diseases = len(self.disease_encoder.classes_)
        logging.info(f"Disease types: {self.disease_encoder.classes_}")

        
        if scaler is None or scaler =="standard":
            self.scaler = StandardScaler()
            self.normalized_data = self.scaler.fit_transform(expression_data)
        else:
            self.scaler = scaler
            self.normalized_data = expression_data

    def __len__(self):
        return len(self.expression_data)
    
    def __getitem__(self, idx):
        return {
            'expression': torch.FloatTensor(self.normalized_data[idx]),
            'disease': self.encoded_diseases[idx],  # Returns scalar int
        }


        



