import os
import pandas as pd
from typing import Dict, Any
from abc import ABC, abstractmethod

def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AbstractDataGenerator(ABC):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        self.config = config
        self.split_no = split_no

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def generate_for_type(self):
        pass


    @abstractmethod
    def save_synthetic_data(self, 
                            synthetic_features: pd.DataFrame, 
                            synthetic_labels: pd.DataFrame, 
                            experiment_name: str):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def load_from_checkpoint(self):
        pass


class BaseDataGenerator(AbstractDataGenerator):
    def __init__(self, config: Dict[str, Any], split_no: int = 1):
        super().__init__(config, split_no)
        self.dataset_config = self.config["dataset_config"]

        # Common configuration loading logic
        self.generator_name = self.config.get('generator_name')
        config_key = f"{self.generator_name}_config"

        if config_key in config:
            self.generator_config = config[config_key]
        else:
            raise ValueError(f"No config found for generator: {self.generator_name}")

        self.dataset_name = self.dataset_config["name"]
        self.subtype_col_name = self.dataset_config["subtype_col_name"]

    def save_synthetic_data(self, 
                            synthetic_features: pd.DataFrame, 
                            synthetic_labels: pd.DataFrame,
                            experiment_name: str = ""):
        data_save_dir = os.path.join(self.config["dir_list"]["home"],
                                     self.config["dir_list"]["data_save_dir"])
        
        syn_save_dir = os.path.join(
                                    data_save_dir, 
                                    self.dataset_name, 
                                    "synthetic", 
                                    self.generator_name,
                                    experiment_name
                                    )
        check_dirs(syn_save_dir)
                
        # save synthetic features and labels
        synthetic_features.to_csv(os.path.join(
            syn_save_dir, f"synthetic_data_split_{self.split_no}.csv"), index=False)
        synthetic_labels.to_csv(os.path.join(
            syn_save_dir, f"synthetic_labels_split_{self.split_no}.csv"), index=False)

        print(f"Synthetic data saved in {syn_save_dir}.")

    @abstractmethod
    def generate(self):
        """Generate synthetic data based on your configuration."""
        pass

    @abstractmethod
    def generate_for_type(self):
        """Generate synthetic data per subtype/type based on your configuration."""
        pass


    @abstractmethod
    def train(self):
        """Train your generator model."""
        pass

    @abstractmethod
    def load_from_checkpoint(self):
        """Load your model from a checkpoint."""
        pass

