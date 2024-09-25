import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
  


from scipy.integrate import simpson

list_colors = [
    "palevioletred",
    "royalblue",
    "chartreuse",
    "teal",
    "burlywood",
    "lightpink",
    "dodgerblue",
    "midnightblue",
    "lime",
    "lightsteelblue",
    "peru",
    "orchid",
    "steelblue",
    "lavender",
    "cadetblue",
    "mintcream",
    "lightseagreen",
    "coral",
    "darkslateblue",
    "indigo"
]




def plot_size_distributions(bin_centers, list_concentrations, savepath, name, list_legend_labels=None, title=None):

    fig, ax = plt.subplots(2,1, figsize=(16,16), sharex=True, sharey=True)

    for i in range(len(list_concentrations)):
        if list_legend_labels is not None and len(list_legend_labels)<15:
            ax[0].plot(bin_centers, list_concentrations[i], color=list_colors[i], alpha=0.7, label=list_legend_labels[i])
        else:
            ax[0].plot(bin_centers, list_concentrations[i], color='royalblue', alpha=0.7)

    concentrations_matrix = np.array(list_concentrations).T
    average_concentration = np.mean(concentrations_matrix, axis=1)
    ax[1].plot(bin_centers, average_concentration, color='darkblue', label='Average concentration')
    
    std_concentration = np.std(concentrations_matrix, axis=1)
    
    lower = average_concentration - std_concentration
    upper = average_concentration + std_concentration
    ax[1].fill_between(x=bin_centers, y1=lower, y2=upper, color='darkblue', alpha=0.1, label='Standard deviation')
   
    ax[1].legend(fontsize=13)
    if list_legend_labels is not None:
        ax[0].legend(fontsize=13)
        
    if title is not None:
        fig.suptitle(title, fontsize=20)
    
    for k in range(2):
        ax[k].set_ylabel('Concentration (particles/mL)', fontsize=15)
        ax[k].set_xlabel('Size (nm)', fontsize=15)
        ax[k].tick_params(axis='both', labelsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, name+'_size_distributions.png'))
    plt.close(fig)




def barplot(list_data, list_legend_labels, name, savepath, title=None):

    fig, ax = plt.subplots(1, figsize=(20,13), sharex=True)

    sorted_indexes = np.argsort(np.array(list_data))
    sorted_data = np.array(list_data)[sorted_indexes]
    sorted_labels = np.array(list_legend_labels)[sorted_indexes]
    
    ax.bar(x=np.arange(len(sorted_labels)), height=sorted_data, color='royalblue')
    ax.set_xticks(np.arange(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, fontsize=18, rotation=45, rotation_mode="anchor", ha="right")
    
    if title is not None:
        ax.set_title(title, fontsize=20)

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, name+'_barplot.png'))
    plt.close(fig)