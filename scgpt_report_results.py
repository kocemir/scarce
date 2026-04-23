import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import numpy as np
import pandas as pd
# Define a basic feedforward neural network
class BasicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__=="__main__":
    

    """
        # I nstantiate the model
    input_size = 10
    hidden_size = 20
    output_size = 1
    model = BasicModel(input_size, hidden_size, output_size)

    # Save the entire model
    torch.save(model, 'basic_model.pth')

    # Load the entire model
    loaded_model = torch.load('basic_model.pth')
    print(loaded_model)

    """
    acc_list=[]
    for i in [1,2,3,0,5,6,7,8,9]:
        file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_ms_run_{i}/results.pkl")
        with open(file_path, "rb") as file:
            results= pickle.load(file)
       
            acc_list.append(100*results["results"]["test/macro_f1"]) #macro_f1
            
            print(results["seed_numbers"])
            
    file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_ms_median/results.pkl")
    with open(file_path, "rb") as file:
            results= pickle.load(file)
            acc_list.append(100*results["results"]["test/macro_f1"])
            

    acc_np= np.array(acc_list)  
    print("All accuracy:", acc_np)
    print("Median:",np.median(acc_np))
    print("Average:",np.mean(acc_np))
    print("Standard Deviation", np.std(acc_list))
    print("Seed numbers:", results["seed_numbers"])

    
    # Load DataFrame from a CSV file
    loaded_df = pd.read_csv('/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/results_csv/results_ms_type3.csv', index_col=0)  # Ensure the first column is used as the DataFrame index

    # Print the loaded DataFrame
    print(loaded_df)
    



