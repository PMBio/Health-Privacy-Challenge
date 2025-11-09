import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotnine import ggplot, aes, geom_point, labs, scale_color_manual

class Plotting:
    #@staticmethod
    #def perform_pca(data):
    #    pca = PCA(n_components=2)
        ## standardize here 
    #    principal_components = pca.fit_transform(data)
    #    return pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    @staticmethod
    def perform_pca(data, scaler=None, pca_model=None):
        """
        Perform PCA on data.

        Args:
            data (np.array or pd.DataFrame): features to reduce
            scaler (StandardScaler, optional): pre-fitted scaler
            pca_model (PCA, optional): pre-fitted PCA model

        Returns:
            pd.DataFrame: 2D PCA components
        """
        # Fit scaler if not provided
        if scaler is None:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
        else:
            data_scaled = scaler.transform(data)

        # Fit PCA if not provided
        if pca_model is None:
            pca_model = PCA(n_components=2)
            principal_components = pca_model.fit_transform(data_scaled)
        else:
            principal_components = pca_model.transform(data_scaled)

        return pd.DataFrame(data=principal_components, columns=['PC1', 'PC2']), scaler, pca_model


    @staticmethod
    def plot_pca_and_save(original_pca, synthetic_pca, save_file):
        original_pca['Type'] = 'Original'
        synthetic_pca['Type'] = 'Synthetic'

        combined_pca = pd.concat([original_pca, synthetic_pca], ignore_index=True)

        plot = (ggplot(combined_pca, aes(x='PC1', y='PC2', color='Type'))
                + geom_point(alpha=0.5)
                + labs(title='PCA of Original and Synthetic Data',
                       x='Principal Component 1',
                       y='Principal Component 2')
                + scale_color_manual(values=['blue', 'orange']))
        
        plot.save(save_file)

        #return plot
