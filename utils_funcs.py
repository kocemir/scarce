import pickle
import random
import os
from pathlib import Path

import numpy as np
import torch
from dataset_graph import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import sys
sys.path.insert(0, "../")
from scgpt import prepare_dataloader

"""

GC: #gene x #cell
CC: #cell x #cell
CG: #cell x #gene
GG: #gene x # gene

"""


def set_seeds(seed_no: int = 42):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  #this was originall true.But  you should set it to False to guarantee super reprodcubility in your code!!!!!



def compute_metrics(output, labels):  
    preds = output.max(1)[1].type_as(labels)
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    w_f1 = f1_score(y_true, y_pred, average="weighted")
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    prec= precision_score(y_true,y_pred,average="macro",zero_division=0)
    recall=recall_score(y_true,y_pred,average="macro",zero_division=0)
    return {"w_f1": w_f1, "macro": macro, "micro": micro, "acc": acc,"precision":prec,"recall":recall}


# Match type_run / trainers (FlashAttention needs CUDA for this checkpoint).
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_loaders(dataset_name,batch_size):

    loader_list=[]

    train_data_dict= torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/train_loader.pth") 
    valid_data_dict= torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/valid_loader.pth")
    test_data_dict= torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/test_loader.pth")
  
    train_loader= prepare_dataloader(train_data_dict, batch_size=batch_size)
    valid_loader =  prepare_dataloader(valid_data_dict,batch_size=batch_size)
    test_loader= prepare_dataloader(test_data_dict,batch_size=batch_size)
    loader_list.append(train_loader)
    loader_list.append(valid_loader)
    loader_list.append(test_loader)
   
    return loader_list


def get_encoder_outputs(dataset_name):
    emb_path = f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/model_embeddings_{dataset_name}.pt" 
    cls_logits_path = f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/model_logits_{dataset_name}.pt"
    
    
    ## These two are embeddings and logits of scBERT. This is temporarily here for ablation study. When all the ablation study zımbırtısı is finished, comment here.
    #emb_path = f"/auto/k2/aykut3/scbert/scBERT/ablation_results/model_embeddings_{dataset_name}_ordered.pt" 
    #cls_logits_path = f"/auto/k2/aykut3/scbert/scBERT/ablation_results/model_logits_{dataset_name}_ordered.pt"
  
    x = torch.load(emb_path)
    cls_logits = torch.load(cls_logits_path)
    print(cls_logits.size())
  
    

    return x.to(device), cls_logits.to(device)



def get_variables(model_type: str, path, dataset: Dataset):
    unit_gene = torch.eye(dataset.GG.shape[0]).to(device)
    unit_cell= torch.eye(dataset.CC.shape[0]).to(device)
    name = dataset.dataset_name
    if model_type == "type1":
        if path == "GG-CG":
            x = unit_gene
            cls_logit = None
            fan_in = dataset.GG.shape[1]
            update_cls = False
        elif path == "CG-CC":  # GC-CG
            x = unit_gene
            cls_logit = None
            fan_in = dataset.CG.shape[1]
            update_cls = False
        elif path == "GC-CG":
            x = unit_cell
            cls_logit = None
            fan_in = dataset.GC.shape[1]
            update_cls = False
        elif path == "CC-CC":
            x = unit_cell
            cls_logit = None
            fan_in = dataset.CC.shape[1]
            update_cls = False
        else:
            raise ValueError("Path must be one of GG-CG,CG-CC, GC-CG, CC-CC")
            
    elif model_type == "type2":
        if path == "GG-CG":
            x = unit_gene
            cls_logit = None
            fan_in = dataset.GG.shape[1]
            update_cls = False
        elif path == "GC-CG":  # XT-X
            x = get_encoder_outputs(name)[0]
            cls_logit = None
            fan_in = x.size(1)
            update_cls = False
        elif path == "CG-CC":  # X-N
            x = unit_gene
            cls_logit = None
            fan_in = dataset.CG.shape[1]
            update_cls = False
        elif path == "CC-CC":
            x = get_encoder_outputs(name)[0]
            cls_logit = None
            fan_in = x.size(1)
            update_cls = False
        else:
           raise ValueError("Path must be one of GG-CG,CG-CC, GC-CG, CC-CC")
           
    elif model_type == "type3":
        if path == "GG-CG":
            x = unit_gene
            cls_logit = get_encoder_outputs(name)[1]
            fan_in = dataset.GG.shape[1]
            update_cls = False
        elif path == "GC-CG":
            x, cls_logit = get_encoder_outputs(name)
            fan_in = x.size(1)
            update_cls = False
        elif path == "CG-CC":
            x = unit_gene
            cls_logit = get_encoder_outputs(name)[1]
            fan_in = dataset.CG.shape[1]
            update_cls = False
        elif path == "CC-CC":
            x, cls_logit = get_encoder_outputs(name)
            fan_in = x.size(1)
            update_cls = False            
        else:
            raise ValueError("Path must be one of GG-CG,CG-CC, GC-CG, CC-CC")

            
    elif model_type =="type4":
         if path == "GG-CG":
            x= unit_gene
            cls_logit= get_encoder_outputs(name)[1]
            fan_in = dataset.GG.shape[1]
            update_cls = False
         elif path == "GC-CG":
            x, cls_logit = get_encoder_outputs(name)
            n_cell = dataset.CC.shape[0]
            x = x[:n_cell, :].to(device)
            cls_logit = cls_logit[:n_cell, :].to(device)
            fan_in = x.size(1)
            update_cls = True
         elif path == "CG-CC":
            x=unit_gene
            cls_logit=get_encoder_outputs(name)[1]
            fan_in = dataset.GG.shape[1]
            update_cls = False
         elif path == "CC-CC":
            x, cls_logit = get_encoder_outputs(name)
            n_cell = dataset.CC.shape[0]
            x = x[:n_cell, :].to(device)
            cls_logit = cls_logit[:n_cell, :].to(device)
            fan_in = x.size(1)
            update_cls = True
         else:
            raise ValueError("Path must be one of GG-CG,CG-CC, GC-CG, CC-CC")

    return x, cls_logit, fan_in, update_cls


def get_A_s(dataset: Dataset, path):
    if path == "GG-CG":
        return [dataset.GG.to(device), dataset.CG.to(device)]
    elif path == "CG-CC":
        return [dataset.CG.to(device), dataset.CC.to(device)]
    elif path == "CC-CC":
        return [dataset.CC.to(device),dataset.CC.to(device)]
    elif path == "GC-CG":
        return [dataset.GC.to(device), dataset.CG.to(device)]
    else:
         raise ValueError("Path must be one of GG-CG,CG-CC, GC-CG, CC-CC")

"""
def get_A_s(dataset: Dataset, path):
    adj_list = []
    for layer in path:
        if layer == "GG":
            adj_list.append(dataset.GG.to(device))
        elif layer == "GC":
            adj_list.append(dataset.GC.to(device))
        elif layer == "CG":
            adj_list.append(dataset.CG.to(device))
        elif layer == "CC":
            adj_list.append(dataset.CC.to(device))
        else:
            raise ValueError(f"Invalid layer combination: {layer}")        

    return adj_list  
"""

def results_dict():
    return {
        "type":[],
        "dataset":[],
        "path": [],
        "test_acc": [],
        "test_recall":[],
        "test_precision":[],
        "test_f1": [],
        "test_preds":[],
        "test_true":[],
        "avg_epoch_time": [],
        "best_test_acc": [],
        "best_test_recall": [],
        "best_test_precision": [],
        "best_test_f1": [],
        "best_test_epoch": [],
        "best_test_preds": [],
        "best_test_true": [],
    }




if __name__=="__main__":   
   x,logits= get_encoder_outputs("ms")
   print(x.size())
   print(logits.size())
   # Print the device of the tensor
   print("Device x:", x.device)
   print("Device logits:", logits.device)
   loaders=get_loaders("ms",32)


  