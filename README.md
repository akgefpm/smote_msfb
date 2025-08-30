
## SMOTE-MSFB: A New Oversampling Technique for Sparse Binary Data  

We introduce **SMOTEN-Minority focused Select Features for Binary data (SMOTE-MSFB)**,  
a novel resampling method designed for high-dimensional, sparse binary datasets.  

### 🔑 Key Highlights
- **Feature-aware neighborhoods:** Uses **mutual information (MI)** to select the most relevant features, then defines minority neighborhoods with a **weighted Jaccard distance**.  
- **Focused resampling:** Employs a **logistic regression filter** to target only those minority samples near the decision boundary, avoiding points deep in minority or majority regions that add little value.  
- **Union-based synthesis:** Generates synthetic samples by combining (“union voting”) feature activations across neighbors, which increases signal strength in sparse binary spaces.  
- **Separation of tasks:** Feature selection for classification is performed **after** resampling, since the optimal features for generating synthetic samples may differ from those needed for classification.  

This approach improves the quality of synthetic samples, reduces noise,  
and enhances classifier performance on imbalanced binary datasets.  


<img width="1036" height="487" alt="image" src="https://github.com/user-attachments/assets/e321de38-ae4f-44d3-8fdc-c7cf7f26eea8" />
