from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def find_proximity_cont(cfe, original_datapoint, continuous_features, mads):
    diff = original_datapoint[continuous_features].to_numpy() - cfe[continuous_features].to_numpy()
    dist_cont = np.mean(np.divide(np.abs(diff), mads))
    sparsity_cont = diff[0].nonzero()[0].shape[0]
    return dist_cont, sparsity_cont

def find_proximity_cat(cfe, original_datapoint, encoded_columns):
    """
    cfe: Counterfactual example after one-hot encoding.
    original_datapoint: Original data point after one-hot encoding.
    encoded_columns: List of columns after one-hot encoding.
    """
    sparsity_cat = 0
    total_features = len(encoded_columns)

    diff = original_datapoint[encoded_columns].to_numpy() - cfe[encoded_columns].to_numpy()
  
    sparsity_cat = np.sum(np.any(diff != 0, axis=0))

    dist_cat = sparsity_cat * 1.0 / total_features
    return dist_cat, sparsity_cat

def get_encoded_columns(original_column, cfe_columns):
    """Get the OneHotEncoded column names for a given original column."""
    return [col for col in cfe_columns if col.startswith(original_column + "_")]

def follows_causality(cfe, original_datapoint, immutable_features, 
        non_decreasing_features, correlated_features):
    follows = True
    diff = cfe.to_numpy().astype(float) - original_datapoint.to_numpy()
    m2 = (diff != 0)[0].nonzero()
    changed_columns = cfe.columns[m2].tolist()
    
    # Convert immutable features to their encoded names
    immutable_features_encoded = []
    for feature in immutable_features:
        immutable_features_encoded.extend(get_encoded_columns(feature, cfe.columns))
    
    # Check immutable features
    changed_immutable_features = set(changed_columns).intersection(set(immutable_features_encoded))
    if changed_immutable_features:
            follows = False
            return follows

    # Convert non-decreasing features to their encoded names
    non_decreasing_features_encoded = []
    for feature in non_decreasing_features:
        non_decreasing_features_encoded.extend(get_encoded_columns(feature, cfe.columns))
    
    # Check non-decreasing features
    diff_nondecrease = cfe[non_decreasing_features_encoded].to_numpy().astype(float) - original_datapoint[non_decreasing_features_encoded].to_numpy()
    m2 = (diff_nondecrease < 0)[0].nonzero()
    if m2[0].shape[0] > 0:
        follows = False
        return follows
    
    # Check correlated features
    for f1, f2 in correlated_features:
        f1_encoded = get_encoded_columns(f1, cfe.columns)
        f2_encoded = get_encoded_columns(f2, cfe.columns)
        for col1 in f1_encoded:
            seq_f1 = cfe.columns.tolist().index(col1)
            for col2 in f2_encoded:
                seq_f2 = cfe.columns.tolist().index(col2)
                if (diff[0][seq_f1] > 0 and diff[0][seq_f2] <= 0) or (diff[0][seq_f1] < 0 and diff[0][seq_f2] >= 0):
                    follows = False
                    return follows
 

    return follows




def find_manifold_dist(cfe, knn):
    nearest_dist, nearest_points = knn.kneighbors(cfe.to_numpy(), 1, return_distance=True)
    quantity = np.mean(nearest_dist)		# Now we have that lambda, so no need to divide by self.no_points 
    return quantity

def calculate_metrics(counterfactuals_list, dataset, continuous_features, categorical_features,scaler, method, target_name):
    
     # 获取OneHotEncoder的输出列名
    encoder = scaler.named_transformers_['cat']
    #categorical_features = dataset.columns[categorical_ids].tolist()
    encoded_columns = encoder.get_feature_names_out(input_features=categorical_features).tolist()
    
    # 组合连续特征和编码后的分类特征的列名
    all_transformed_columns = continuous_features + encoded_columns
    
    # 1. Transform the dataset
    transformed_dataset = scaler.transform(dataset.drop(columns=[target_name]))
    transformed_dataset = pd.DataFrame(transformed_dataset, columns=all_transformed_columns)
    
    # 2. Calculate normalized MADS for continuous features
    normalized_mads = {}
    for feature in continuous_features:
        normalized_mads[feature] = np.median(abs(transformed_dataset[feature].values - np.median(transformed_dataset[feature].values)))
    mads = [normalized_mads[key] if normalized_mads[key] != 0.0 else 1.0 for key in normalized_mads]
    
    # 3. Setup KNN for manifold distance
    knn = NearestNeighbors(n_neighbors=5, p=1)
    knn.fit(transformed_dataset)
    
    # 4. Loop through counterfactuals to calculate metrics
    avg_proximity_cont = []
    avg_proximity_cat = []
    avg_sparsity = []
    # avg_causality = []
    # avg_manifold_dist = []
    
    
    for instance_series, cf_df in counterfactuals_list:
        instance_df = pd.DataFrame(instance_series)
    
        if method == 'cfrl':
            instance_df = instance_df.iloc[:, :-1]
            cf_df = cf_df.iloc[:, :-1]
    
        elif method == 'kdtree':
            instance_df = instance_df.T
    
        elif method in ['random', 'prototype']:
            #instance_df = instance_df.T
            cf_df = cf_df.iloc[:, :-1]
        elif method == "nice":
            cf_df = cf_df
        
        # Transform data
        cfe_transformed = scaler.transform(cf_df)
        instance_df_transformed = scaler.transform(instance_df)
        cfe_transformed = pd.DataFrame(cfe_transformed, columns=all_transformed_columns)
        instance_df_transformed = pd.DataFrame(instance_df_transformed, columns=all_transformed_columns)

        

        # Proximity
        proximity_cont, sparsity_cont = find_proximity_cont(cfe_transformed, instance_df_transformed, continuous_features, mads)
        proximity_cat, sparsity_cat = find_proximity_cat(cfe_transformed, instance_df_transformed, encoded_columns)
        
        # Sparsity
        sparsity = sparsity_cont + sparsity_cat
        
        # # Causality
        # causality = follows_causality(cfe_transformed, instance_df_transformed, immutable_features, 
        #                               non_decreasing_features, correlated_features)
        
        # # Manifold Distance
        # manifold_dist = find_manifold_dist(cfe_transformed, knn)
        
        avg_proximity_cont.append(proximity_cont)
        avg_proximity_cat.append(proximity_cat)
        avg_sparsity.append(sparsity)
        # avg_causality.append(causality)
        # avg_manifold_dist.append(manifold_dist)
    
    # 5. Calculate average metrics
    avg_metrics = {
        "avg_proximity_cont": np.mean(avg_proximity_cont),
        "avg_proximity_cat": np.mean(avg_proximity_cat),
        "avg_sparsity": np.mean(avg_sparsity),
        # "avg_causality": np.mean(avg_causality),
        # "avg_manifold_dist": np.mean(avg_manifold_dist)
    }
    
    return avg_metrics
