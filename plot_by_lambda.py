import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

cur_dir= os.getcwd()
base_dir=os.path.join(cur_dir,"scgnn_lambda")


type_name="type3"
dataset_name="ms"



def load_and_plot_results(base_dir, dataset_name, type_name):
    path_list= ["GG-CG","GC-CG","CG-CC","CC-CC"]
    lambda_list = list(np.flip(np.array([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0])))
    
    ### Added the new LateX path names as a list for the legend, and constrained the legend to bottom-left side of the plot, date is 30/07/2024.
    pathway_list = ['$A_{GG}-A_{CG}$', '$A_{GC}-A_{CG}$', '$A_{CG}-A_{CC}$', '$A_{CC}-A_{CC}$', '$scGPT$']
        # Define pathway list with mathbf style
    pathway_list = [
        r'$P1:$ $\mathbf{A_{GG}-A_{CG}}$',
        r'$P2:$ $\mathbf{A_{GC}-A_{CG}}$',
        r'$P3:$ $\mathbf{A_{CG}-A_{CC}}$',
        r'$P4:$ $\mathbf{A_{CC}-A_{CC}}$'
    ]

    
    #plt.figure(figsize=(6,6))
    
    acc_list= np.zeros((len(lambda_list),len(path_list)))
    std_list= np.zeros((len(lambda_list),len(path_list)))
    
    for i,lm in enumerate(lambda_list):
        lm= f"lamda_{str(lm)}"
        for j, pt in enumerate(path_list):
                    
            load_path = os.path.join(base_dir, dataset_name,type_name,lm,pt)
            try:
                final_results_file = os.path.join(load_path, os.listdir(load_path)[-1])
            except IndexError:
                continue  # Skip if no files found in the directory

            with open(final_results_file, "rb") as f:
                loaded_results = pickle.load(f)

            acc_list[i,j]= np.mean(np.array(loaded_results['test_acc']))
            std_list[i,j]= np.std(np.array(loaded_results['test_acc']))
    
        # Number of columns
    num_columns = acc_list.shape[1]

    # Different format strings for each column
    formats = ['-o', '--s', '-.^', ':d']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    
    scgpt_acc=acc_list[0,0]*np.ones((len(lambda_list),len(path_list)))
 
    rc("font", **{"size":13.5})
    rc("text", usetex=False)   
    
    # Plot each column
    for i in range(num_columns):
        plt.errorbar(range(len(acc_list)), acc_list[:, i], fmt=formats[i], capsize=5, label=pathway_list[i])
        #plt.plot(range(len(acc_list)), acc_list[:, i], formats, label=pathway_list[i])
    plt.plot(scgpt_acc[:,0],label="scGPT",linestyle="-", color="purple")
    
    # Bold the frame yerr=std_list[:, i],
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Set the desired linewidth

    from matplotlib.font_manager import FontProperties
    font_properties = FontProperties()
    font_properties.set_family('serif')
    font_properties.set_weight('bold')
        
    plt.grid(True)
    # Adjust layout to prevent overlap
  
    plt.legend(pathway_list, loc="lower left")
    # Set the x-ticks
    x_ticks = range(len(lambda_list))
    plt.xticks(ticks=x_ticks, labels=lambda_list,  fontweight="bold", fontsize=12.5)
    
    plt.yticks(fontweight="bold", fontsize=12.5 )
    plt.xlabel(r'Lambda ($\lambda$)',fontsize=15,labelpad=1,fontweight="bold")
    plt.title(dataset_name.upper(),fontweight="bold", fontsize=17.5)
    # Bold the legend labels
    plt.legend(prop={'weight': 'bold'}, loc='lower left')
    plt.ylabel(r'Test Accuracy ($\%$)',fontsize=15,labelpad=1, fontweight="bold")
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"./lambda_plot/{dataset_name}_acc.png",dpi=500)

load_and_plot_results(base_dir, dataset_name, type_name)