
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_resample(x_train, y_train, x_train_upd, y_train_upd, path, version):
    
    # Fit TSNE on the combined data
    tsne = TSNE(n_components=2, random_state=42)
    x_train_upd_tsne = tsne.fit_transform(x_train_upd)
    
    y_train_upd[ len(y_train) :] = 2

    # Defining colors for the classes
    colors = {0: 'lightblue', 1: '#FF0000', 2: 'yellow'}

    for cls in np.unique(y_train_upd):
        indices = np.where(y_train_upd == cls)
        plt.scatter(x_train_upd_tsne[indices, 0], x_train_upd_tsne[indices, 1], c=colors[cls], label=f'Class {cls}', marker='o', alpha=0.7)
    plt.title(version +' - Resampled Data visualization')

    # Adding a legend for the specific class descriptions
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=10, label='Minority Class Sample'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Majority Class Sample'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Regenerated Sample')
    ]

    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.2, -0.1), ncol=1, title='Sample Types', frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(path +"/Rsmpl_"+ version +"_Vzl.jpg")    