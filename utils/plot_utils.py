# functions used to generate plots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def hist_class_distribution(set_x, ax, commands):

    frequencies = []
    
    for command in commands:
        frequencies.append(set_x[set_x.label==command].shape[0])
        
    frequencies = np.array(frequencies)/len(set_x)
    
    ax.bar(commands, frequencies, edgecolor='black', alpha=0.5, color='forestgreen')
    
    
def plot_features(features, title=None):

    _, ax = plt.subplots(figsize=(14, 10))
    ax = sns.heatmap(features)
    ax.set_xlabel("Window", fontsize=18)
    ax.set_ylabel("Features", fontsize=18)

    if title is not None:
        ax.set_title(title, fontsize=20)
            
    plt.tight_layout()
    plt.show()
    
    
def plot_history(history, columns=['loss']):
    
    _, axes = plt.subplots(len(columns), 1, figsize=(8, 5*len(columns)))

    for i, column in enumerate(columns):
        ax = axes[i] if len(columns) > 1 else axes
        ax.plot(history.history[column], label='training', color='blue', linewidth=1.5)
        ax.plot(history.history['val_'+column], label='validation', color='firebrick', linewidth=1.5)
        ax.set_xticks(range(len(history.history['loss'])))
        ax.set_xticklabels(range(1, len(history.history['loss'])+1))
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.5)
        ax.set_ylabel(column)
        ax.legend(edgecolor='black', facecolor='linen', fontsize=12, loc ='best') 

    plt.tight_layout()
    plt.show()
    

def plot_confusion_matrix(cm, labels, annot=True, cmap='Blues', normalize=False, saveit=False, model_name=''):
    _, ax = plt.subplots(figsize=(21, 18))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax = sns.heatmap(cm, vmin=0, vmax=1, annot=annot, xticklabels=labels, yticklabels=labels, cmap=cmap, fmt=".2f")
        
    else:
        ax = sns.heatmap(cm, annot=annot, xticklabels=labels, yticklabels=labels, cmap=cmap)
    
    ax.set_xlabel('Predicted label', fontsize=15)
    ax.set_ylabel('True label', fontsize=15)
    ax.set_title('Confusion matrix', fontsize=18)

    plt.tight_layout()
    
    if saveit:
        plt.savefig(f'figures/cm_{model_name}.png')
    plt.show()