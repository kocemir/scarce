from pathlib import Path
import pickle
import torch
from dataset_graph import Dataset, load_processed_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from configs import EncoderConfig, SearchParams, Type3Config, Type4Config, Type12Config, TypeInput, Type4Input
from graph_models import Type3, Type12, Type4
#from trainers import TypeInput, TypeTrainer, Type4Trainer
from trainers_wportion2 import TypeInput, TypeTrainer, Type4Trainer

from utils_funcs import get_A_s, get_variables,get_loaders,set_seeds,results_dict


from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import gc


import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#sdevice = torch.device('cpu')

def run_type12(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    print(f"Number of unique classes is {n_class}")
    x, _, fan_in, _ = get_variables(model_type, path, data)
    config = Type12Config(fan_in=fan_in, fan_mid=params.fan_mid, fan_out=n_class, dropout=params.gcn_p)
    model = Type12(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gcn_lr, weight_decay=params.wd)
    t_input = TypeInput(x, get_A_s(data, path), data.y.to(device), data.train_ids, data.test_ids, data.valid_ids) 
    trainer = TypeTrainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience)

    return trainer, model


def run_type3(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    print(f"Number of unique classes is {n_class}")
    x, cls_logit, fan_in, _ = get_variables(model_type, path, data)
    gcn_config = Type12Config(fan_in=fan_in, fan_mid=params.fan_mid, fan_out=n_class, dropout=params.gcn_p)
    config = Type3Config(type12_config=gcn_config, cls_logit=cls_logit, lmbd=params.lmbd)
    model = Type3(config).to(device)   
    optimizer = torch.optim.Adam(model.parameters(), lr=params.gcn_lr, weight_decay=params.wd)
    t_input = TypeInput(x, get_A_s(data, path), data.y.to(device), data.train_ids, data.test_ids, data.valid_ids)  
    trainer = TypeTrainer(model, optimizer, t_input)
    trainer.pipeline(params.max_epochs, params.patience)

    return trainer, model



def run_type4(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    print(f"Number of unique classes is {n_class}")
    x, cls_logits , fan_in, update_cls = get_variables(model_type, path, data)
    gcn_config = Type12Config(fan_in=fan_in, fan_mid=params.fan_mid, fan_out=n_class, dropout=params.gcn_p)
    encoder_config = EncoderConfig(model_name="scgpt", dataset_name=dataset_name, n_class=n_class, CLS=True, dropout=params.encoder_p)
    config = Type4Config(type12_config=gcn_config, encoder_config=encoder_config, lmbd=params.lmbd,batch_size=params.batch_size)
    model = Type4(config, path).to(device)
    loaders= get_loaders(dataset_name, config.batch_size)
    
  
    
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": params.encoder_lr},
            {"params": model.gcn.parameters(), "lr": params.gcn_lr},
        ],
        weight_decay=params.wd,
    )
    
    t_input = Type4Input(x=x, A_s=get_A_s(data, path), loaders=loaders, train_ids=data.train_ids, valid_ids=data.valid_ids, test_ids=data.test_ids,y=data.y)
    trainer = Type4Trainer(model, optimizer, t_input, update_cls=update_cls, train_fraction=params.train_fraction)
    trainer.pipeline(params.max_epochs, params.patience)

    return trainer, model

   
def run_type(dataset_name: str, model_type: str, path: str, params: SearchParams):
    if model_type == "type1" or model_type == "type2":
        return run_type12(dataset_name, model_type, path, params)
    elif model_type == "type3":
        return run_type3(dataset_name, model_type, path, params)
    elif model_type == "type4":
        return run_type4(dataset_name, model_type, path, params)
    else:
        raise ValueError("Undefined type!")

 

if __name__=="__main__":
   
    cur_dir=  os.getcwd()
    save_dir= cur_dir+"/scarce_merged"
    if not os.path.exists(path=save_dir):
        save_dir= os.path.join(cur_dir, 'scarce_merged')
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}.")
    else:
         print(f"{save_dir} is already created.")
   
   

    dataset_name="ms"
    type_name="type4"

    path_list= ["GG-CG","GC-CG","CG-CC","CC-CC"]
    path_list= ["CC-CC"]
 
    
    
    params = SearchParams(
        fan_mid=256,
        lmbd=0.7, # default 0.7
        gcn_lr=0.0001, # Original gcn_lr=0.0001
        gcn_p=0.2,
        wd=0.00005,
        patience=10,
        max_epochs=50, #25, 3000   # if batch_size is 32, then use max_epochs as 50
        encoder_lr=0.00001, train_fraction=0.5, # default 1.0
        encoder_p=0.2, batch_size=8 #16 or 32 (used  and reported for the first two paths of ms, then gave up)

    )
    
    
    ##################################################################################
    if dataset_name == "ms":
        data_dir = Path("./data/ms")
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")
        adata.obs["indices"]= np.arange(adata.obs.shape[0])
    
    if dataset_name == "pancreas": #RB
        data_dir = Path("../data/pancreas")
        adata = sc.read(data_dir / "demo_train.h5ad")
        adata_test = sc.read(data_dir / "demo_test.h5ad")
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"    
        data_is_raw = False
        filter_gene_by_counts = False   
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")
        adata.obs["indices"]= np.arange(adata.obs.shape[0])

    if dataset_name == "myeloid":
        data_dir = Path("../data/mye/")
        adata = sc.read(data_dir / "reference_adata.h5ad")
        adata_test = sc.read(data_dir / "query_adata.h5ad")
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
        adata_test_raw = adata_test.copy()
        data_is_raw = False
        filter_gene_by_counts = False   
        adata = adata.concatenate(adata_test, batch_key="str_batch")
        adata.obs["indices"]= np.arange(adata.obs.shape[0])
    ##################################################################################
   
    
    #### Take results from thed save transformer model. Do not afraid, I only look up the seed list
    file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/results.pkl")
    with open(file_path, "rb") as file:
            results= pickle.load(file)   
    seed_list=results["seed_numbers"]


    
    
    ##### CREATE DICTIONARY TO SAVE RESULTS
   
    for pt in path_list:
        print(pt)
        d = results_dict()
        d["path"].append(pt)
        d["type"].append(type_name)
        d["dataset"].append(dataset_name)


        load_dir= save_dir+"/"+f"{dataset_name}/{type_name}/{pt}/portion_{int(params.train_fraction*100)}/" 
        if not os.path.exists(path=load_dir):
                    load_dir= os.path.join(cur_dir, load_dir)
                    os.makedirs(load_dir)
                    print(f"Created directory: {load_dir}.")
        else:
                    print(f"{load_dir} is already created.")
            
       
        for i, seed in enumerate(seed_list[0:3]):
                if seed==15:
                     seed=0
                set_seeds(seed)
                trainer,model=run_type(dataset_name= dataset_name, model_type=type_name, path=pt, params=params)
                
                d["test_acc"].append(100*trainer.metrics["test"]["acc"])
                d["test_f1"].append(100*trainer.metrics["test"]["macro"])
                d["test_precision"].append(100*trainer.metrics["test"]["precision"])
                d["test_recall"].append(100*trainer.metrics["test"]["recall"])
                d["avg_epoch_time"].append(trainer.avg_epoch_time)
                d["test_preds"].append(trainer.y_test_preds) # these are numpyed values
                d["test_true"].append(trainer.y_test_true) # these are numpyed values  ( In all of the runs, I mistakenly wrote trainer.y_test_preds here, but do not worry. You can use results["labels"] as trainer.y_test_true

                best_test = trainer.metrics.get("best_test")
                if best_test is not None:
                    d["best_test_acc"].append(100*best_test["acc"])
                    d["best_test_f1"].append(100*best_test["macro"])
                    d["best_test_precision"].append(100*best_test["precision"])
                    d["best_test_recall"].append(100*best_test["recall"])
                    d["best_test_epoch"].append(trainer.metrics.get("best_test_epoch", -1))
                    d["best_test_preds"].append(getattr(trainer, "best_y_test_preds", None))
                    d["best_test_true"].append(getattr(trainer, "best_y_test_true", None))
                    print(f"[seed {seed}] Best Test Acc: {100*best_test['acc']:.3f} @ epoch {trainer.metrics.get('best_test_epoch', -1)}")
                 


                #If you would like to save model results, uncomment here!
                
                
                result_dir= os.path.join(load_dir, f"dname_{dataset_name}_path_[{pt}]_type_{type_name}_seedid_{str(i)}_seed_{seed}_portion_{params.train_fraction}")
                # Save dictionary using pickle
                with open(result_dir, 'wb') as pickle_file:
                    pickle.dump(d, pickle_file)
                
                
                #equal = np.array_equal(results["labels"],trainer.y_test_true)
                #assert equal
                
                # Free up memory
                del trainer
                torch.cuda.empty_cache()
                gc.collect()
        
        print("Accuracy",sum(d["test_acc"])/len(d["test_acc"]))
        print("F1", sum(d["test_f1"])/len(d["test_f1"]))
        print("Accuracy STD", np.std(np.array(d["test_acc"])))
        print("F1 STD", np.std(np.array(d["test_f1"])))

        if len(d["best_test_acc"]) > 0:
            print("Best Accuracy", sum(d["best_test_acc"])/len(d["best_test_acc"]))
            print("Best F1", sum(d["best_test_f1"])/len(d["best_test_f1"]))
            print("Best Accuracy STD", np.std(np.array(d["best_test_acc"])))
            print("Best F1 STD", np.std(np.array(d["best_test_f1"])))
            print("Best Epochs", d["best_test_epoch"])






























       
                        



