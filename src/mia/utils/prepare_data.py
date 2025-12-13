import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class MIADataLoader:
    def __init__(self, 
                synthetic_file: str,
                membership_test_file: str,
                membership_lbl_file:str,
                membership_label_col: str,
                generator_model: str,
                reference_file: str = None):
        self.generator_model = generator_model
        self.membership_test_file = membership_test_file
        self.membership_lbl_file = membership_lbl_file
        self.membership_label_col = membership_label_col
        self.synthetic_file = synthetic_file
        self.reference_file = reference_file


    def load_synthetic_data(self):
        synthetic_data = pd.read_csv(self.synthetic_file).values
        synthetic_data = StandardScaler().fit_transform(synthetic_data)

        synthetic_labels_file = self.synthetic_file.replace("_data_", "_labels_")
        synthetic_labels = pd.read_csv(synthetic_labels_file)

        return synthetic_data, synthetic_labels
    

    
    def load_original_data(self, save_dir, dataset_name, align_to_synthetic=None):
        # get split num
        split_num = self.synthetic_file.split("_")[-1].split(".")[0]
        print(split_num)
        real_save_dir = os.path.join(save_dir, dataset_name, "real")
        original_data = pd.read_csv(os.path.join(
            real_save_dir, f"X_train_real_split_{split_num}.csv"))#.values
        
        ## this is needed for others... 
        if align_to_synthetic is not None:
            common_cols = align_to_synthetic.columns
            X_train_real = X_train_real[common_cols]

        X_train_real = X_train_real.values

        original_data = StandardScaler().fit_transform(original_data)
        original_labels = pd.read_csv(os.path.join(
            real_save_dir, f"y_train_real_split_{split_num}.csv"))

        print(original_data.shape)

        return original_data, original_labels
    
    
    def load_membership_dataset(self, align_to_synthetic=None):
        if not os.path.exists(self.membership_test_file):
            raise FileNotFoundError("Membership test dataset is missing.")
        dataset = pd.read_csv(self.membership_test_file, 
                                         sep="\t", 
                                         index_col=0).T #.values
        

        ## this is needed for others... 
        if align_to_synthetic is not None:
            common_cols = align_to_synthetic.columns
            dataset = dataset[common_cols]

        dataset = dataset.values
        dataset = StandardScaler().fit_transform(dataset)
        
        print(f"Membership test set is loaded. Size {dataset.shape}")
        return dataset
    
    def load_membership_labels(self):
        labels = None
        if self.membership_lbl_file is not None:
            labels = pd.read_csv(self.membership_lbl_file, index_col=0)[
                                        self.membership_label_col].values
            print(f"Membership test labels are loaded. Size {len(labels)}")
            
        return labels



    
    def load_reference_data(self, align_to_synthetic=None):
        if self.reference_file:
            reference = pd.read_csv(self.reference_file, 
                                         sep="\t", 
                                         index_col=0).T
            
            #print(reference.head())
                   ## this is needed for others... 
            if align_to_synthetic is not None:
                common_cols = align_to_synthetic.columns
                reference = reference[common_cols]

            #dataset = dataset.values

            reference = StandardScaler().fit_transform(reference)
            return reference
        else:
            return None
    
    
    @staticmethod
    def save_files(save_dir, file_name_list, array_list):

        assert len(file_name_list) == len(array_list)

        for i in range(len(file_name_list)):
            np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)