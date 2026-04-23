from pathlib import Path
import pickle
import torch
from dataset_graph import Dataset, load_processed_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from configs import EncoderConfig, SearchParams, Type3Config, Type4Config, Type12Config, TypeInput, Type4Input
from graph_models import Type3, Type12, Type4
from trainers import TypeInput, TypeTrainer, Type4Trainer

from utils_funcs import get_A_s, get_variables,get_loaders,set_seeds,results_dict


from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import gc


import numpy as np


# Set the device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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

    return trainer


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

    return trainer



def run_type4(dataset_name: str, model_type: str, path: str, params: SearchParams):
    data = load_processed_dataset(dataset_name)
    n_class = data.y.cpu().view(-1).unique().shape[0]
    print(f"Number of unique classes is {n_class}")
    x, cls_logits , fan_in, update_cls = get_variables(model_type, path, data)
    gcn_config = Type12Config(fan_in=fan_in, fan_mid=params.fan_mid, fan_out=n_class, dropout=params.gcn_p)
    encoder_config = EncoderConfig(model_name="scgpt", dataset_name=dataset_name, n_class=n_class, CLS=True, dropout=params.encoder_p)
    config = Type4Config(type12_config=gcn_config, encoder_config=encoder_config, lmbd=params.lmbd,batch_size=params.batch_size)
    model = Type4(config).to(device)
    loaders= get_loaders(dataset_name, config.batch_size)
    
  
    
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": params.encoder_lr},
            {"params": model.gcn.parameters(), "lr": params.gcn_lr},
        ],
        weight_decay=params.wd,
    )
    
    t_input = Type4Input(x=x, A_s=get_A_s(data, path), loaders=loaders, train_ids=data.train_ids, valid_ids=data.valid_ids, test_ids=data.test_ids,y=data.y)
    trainer = Type4Trainer(model, optimizer, t_input, update_cls=update_cls)
    trainer.pipeline(params.max_epochs, params.patience)

    return trainer

   
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
   
    cur_dir=  os. getcwd()
    save_dir= cur_dir+"/scgnn_lambda"
    if not os.path.exists(path=save_dir):
        save_dir= os.path.join(cur_dir, 'scgnn_lambda')
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}.")
    else:
         print(f"{save_dir} is already created.")
   
   

    dataset_name="myeloid"
    type_name="type3"

    path_list= ["GG-CG","GC-CG","CG-CC","CC-CC"]

    
   
    #### Take results from the save transformer model
    file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/results.pkl")
    with open(file_path, "rb") as file:
            results= pickle.load(file)   
    seed_list=results["seed_numbers"]
    
    lambda_list= [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0] #0.7 is by default performed in earlier experiments
    
    for l in lambda_list:

            params = SearchParams(
                        fan_mid=256,
                        lmbd=l,
                        gcn_lr=0.0001,
                        gcn_p=0.2,
                        wd=0.00005,
                        patience=10,
                        max_epochs=3000,
                        encoder_lr=0.00001,
                        encoder_p=0.2, batch_size=16

            )
                
                ##### CREATE DICTIONARY TO SAVE RESULTS
            
            for pt in path_list:
                    print(pt)
                    d = results_dict()
                    d["path"].append(pt)
                    d["type"].append(type_name)
                    d["dataset"].append(dataset_name)
                
                    for i, seed in enumerate(seed_list):
                            if seed==15:
                                seed=0
                            set_seeds(seed)
                            trainer=run_type(dataset_name= dataset_name, model_type=type_name, path=pt, params=params)
                            d["test_acc"].append(100*trainer.metrics["test"]["acc"])
                            d["test_f1"].append(100*trainer.metrics["test"]["macro"])
                            d["test_precision"].append(100*trainer.metrics["test"]["precision"])
                            d["test_recall"].append(100*trainer.metrics["test"]["recall"])
                            d["avg_epoch_time"].append(trainer.avg_epoch_time)
                            d["test_preds"].append(trainer.y_test_preds) # these are numpyed values
                            d["test_true"].append(trainer.y_test_true) # these are numpyed values
                            
                
                           
                            
                            # These chek and save block can be slided in left to not to save for each seed iteration, last one can be saved normally.
                            # However, it can stay as it is now.
                            load_dir= save_dir+"/"+f"{dataset_name}/{type_name}/lamda_{str(l)}/{pt}/" 
                            if not os.path.exists(path=load_dir):
                                load_dir= os.path.join(cur_dir, load_dir)
                                os.makedirs(load_dir)
                                print(f"Created directory: {load_dir}.")
                            else:
                                print(f"{load_dir} is already created.")
                        

                           
                            result_dir= os.path.join(load_dir, f"dname_{dataset_name}_path_[{pt}]_type_{type_name}_seedid_{str(i)}_seed_{seed}")
                            
                            # Comment out below if you want to dumb model to the directory
                            """
                            # Save dictionary using pickle
                            with open(result_dir, 'wb') as pickle_file:
                                pickle.dump(d, pickle_file)
                            """
                            
                            equal = np.array_equal(results["labels"],trainer.y_test_true)
                            assert equal
                            

                                            # Free up memory
                            del trainer
                            torch.cuda.empty_cache()
                            gc.collect()
























