from pathlib import Path
import anndata
import numpy as np
import os
import pickle 
import torch
import torch.nn.functional as F

from graph_construct import genegene, cellgene, cellcell



class Dataset:

    def __init__(self, dataset_name):
          self.dataset_name= dataset_name  
          data_path = os.path.join("processed_data", self.dataset_name+".h5ad")
          
          self.y, self.train_ids,self.test_ids, self.valid_ids= self.train_test_ids(data_path)
          print("**********",self.train_ids)
          print("**********",self.valid_ids)
          print("**********",self.test_ids)
          print("**********",self.y[self.train_ids][0:10])
          
          
          #self.expression_matrix_binned= self.expression_values(data_path)[0][torch.hstack([self.train_ids,self.valid_ids, self.test_ids])]
          
          self.expression_matrix_binned= self.expression_values(data_path)[0][torch.hstack([self.train_ids,self.valid_ids])]
         
         
          # I created them but I dont use them now.
          self.expression_matrix_raw_all=self.expression_values(data_path)[1] #here, the valid and train is mixed, be carefull!!!
          self.expression_matrix_raw_test= self.expression_matrix_raw_all[self.expression_matrix_raw_all.obs["batch_id"]==1] # consider only the test

          self.generate_graph()

              
    def train_test_ids(self,path):
         
        train_ids, test_ids = [], []

        loaded_ann = anndata.read_h5ad(path)
        all_batch_ids = loaded_ann.obs["batch_id"].tolist()
        for id, val in enumerate(all_batch_ids):
             if val==0:
                  train_ids.append(id)
             if val==1:
                  test_ids.append(id)

        y= loaded_ann.obs["celltype_id"].tolist()
        y= np.array(y)
        

        train_ids= np.array(train_ids)
       
        loaded_data = np.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{self.dataset_name}_median/indices.npz")
        train_indices= train_ids[loaded_data["tr_indices"]]
        valid_indices= train_ids[loaded_data["val_indices"]]

        test_ids= np.array(test_ids)
        
        return torch.tensor(y), torch.tensor(train_indices),torch.tensor(test_ids), torch.tensor(valid_indices)
     
    def expression_values(self,path):
          
          expression_matrix_raw = anndata.read_h5ad(path)
          expression_matrix_binned= expression_matrix_raw.layers["X_binned"]

          return (expression_matrix_binned, expression_matrix_raw)
     
    def generate_graph(self):
         expression_matrix= self.expression_matrix_binned 

         self.GG=genegene(expression_matrix)
         self.CG=cellgene(expression_matrix,n_bins=51)
         self.CC=cellcell(expression_matrix)
         self.GC=self.CG.T

         # Node counts: GG/CC are square on genes / cells; CG is cell–gene bipartite.
         self.n_genes = int(self.GG.shape[0])
         self.n_cells = int(self.CC.shape[0])
         
    
    def __repr__(self) -> str:
        print_str = (
            f"Dataset({self.dataset_name})"
            f"\nTotal  Number of cells: {len(self.expression_matrix_binned)}"
            f"\nTotal  Number of genes: {len(self.expression_matrix_binned[0])}"
            f"\nNumber of unique cell type is: {len(self.y.unique())}"
        )
        if hasattr(self, "n_genes") and hasattr(self, "n_cells"):
            print_str += (
                f"\nGraph nodes — GG (gene–gene): {self.n_genes}, CC (cell–cell): {self.n_cells}"
                f"\nGraph shape — CG: {self.n_cells}×{self.n_genes}, GC: {self.n_genes}×{self.n_cells}"
            )
        return print_str
      
def load_processed_dataset(dataset_name):
    file_path = os.path.join("processed_data", f"{dataset_name}_train.pkl")
    with open(file_path, "rb") as file:
        dataset: Dataset = pickle.load(file)
    return dataset

if __name__ == "__main__":
      
     for datasets in ["ms"]:#,"pancreas","myeloid"]: #["ms","pancreas","myeloid"]
         

        DATA_DIR = os.path.join("processed_data", datasets +"_train.pkl")
         


        if not os.path.exists(DATA_DIR):   
            dataset = Dataset(dataset_name=datasets)
            print(dataset)

            with open(DATA_DIR, "wb") as f:
                    pickle.dump(dataset, f)
            print("Dataset object is pickled.")
        else:
            print(f"{DATA_DIR} already exists.")   
    
     dataset=load_processed_dataset(datasets)
     print(len(dataset.test_ids))
     #print(dataset.expression_matrix_raw_test.obs)
 
    
          
          
     