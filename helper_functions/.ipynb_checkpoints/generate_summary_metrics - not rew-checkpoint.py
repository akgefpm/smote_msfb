import pandas as pd

def generate_summary_metrics(df: pd.DataFrame, path: str, domain: str) -> pd.DataFrame:
    """
    Generate summary metrics for binary classification datasets.
    
    Args:
        df (pd.DataFrame): Input dataset.
        path (str): Output location to save the summary metrics parquet file.
        domain (str): Domain of the dataset (text argument).
    
    Returns:
        pd.DataFrame: Summary metrics dataframe.
    """
    
    # Identify covariate and target columns
    covariate_cols = [col for col in df.columns if col.startswith("f_")]
    target_cols = [col for col in df.columns if col.startswith("target_")]
    
    # Ensure target columns are numeric (convert "0"/"1" -> 0/1)
    df[target_cols] = df[target_cols].apply(pd.to_numeric, errors="coerce")
    
    summary_data = []
    n_rows = len(df)
    n_covariates = len(covariate_cols)
    
    for target in target_cols:
        value_counts = df[target].value_counts().to_dict()
        majority_class = max(value_counts, key=value_counts.get) if value_counts else None
        minority_class = min(value_counts, key=value_counts.get) if value_counts else None
        
        majority_count = value_counts.get(majority_class, 0)
        minority_count = value_counts.get(minority_class, 0)
        
        imbalance_ratio = (majority_count / minority_count) if minority_count > 0 else float("inf")
        
        summary_data.append({
            "target_variable": target,
            "num_rows": n_rows,
            "num_covariates": n_covariates,
            "imbalance_ratio": imbalance_ratio,
            "minority_count": minority_count,
            "majority_count": majority_count,
            "domain": domain
        })
    
    summary_metrics = pd.DataFrame(summary_data)
    
    # Save as parquet file
    summary_metrics.to_parquet(f"{path}/summary_metrics.parquet", index=False, engine="pyarrow")
    
    return summary_metrics
