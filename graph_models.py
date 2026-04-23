import warnings

import torch
import torch.nn as nn
from configs import EncoderConfig, Type3Config, Type4Config, Type12Config
from layers import GraphConvolution
from torch.nn import functional as F
from typing import List
import sys



sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.model_default import TransformerModel

dataset_name="ms"


class scGPTForAnnotation(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.c=config
        model=torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{self.c.dataset_name}_median/model.pt")
        # Load backbone weights from the pretrained (pre-fine-tuning) scGPT checkpoint.
        # strict=False: pretraining-only heads (flag_encoder, mvc_decoder) are ignored,
        # and cls_decoder keys remain "missing" from this load (they'll be freshly
        # re-initialized below so no fine-tuned head weights leak through).
        pretrained_sd = torch.load(
            "/auto/k2/aykut3/Yunus/EEE492/save/scGPT_human/best_model.pt",
            map_location="cuda",
        )
        missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
        print(f"[scGPTForAnnotation] pretrained backbone loaded; "
              f"missing={len(missing)} (expect cls_decoder.*) "
              f"unexpected={len(unexpected)} (expect flag_encoder/mvc_decoder.*)")

        if hasattr(model, "cls_decoder"):
            for m in model.cls_decoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.transformer = model

    def _run_transformer(self, src, values, src_key_padding_mask, batch_labels=None, cls=True):
        """FlashAttention requires FP16 activations; torch.autocast supplies that on CUDA."""
        if src.device.type != "cuda":
            raise RuntimeError(
                "This checkpoint uses FlashAttention (CUDA + FP16 only). "
                "Use a GPU and ensure batches are moved with .to(cuda) in trainers/type_run."
            )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self.transformer(src, values, src_key_padding_mask, batch_labels, cls)

    def forward(self, src, values, src_key_padding_mask, batch_labels=None, cls=True):
        return self._run_transformer(src, values, src_key_padding_mask, batch_labels, cls)

"""
self.c = config
self.gcn1 = GraphConvolution(self.c.fan_in, self.c.fan_mid)
self.ln1 = nn.LayerNorm(self.c.fan_mid)
self.gcn2 = GraphConvolution(self.c.fan_mid, self.c.fan_mid // 2)
self.ln2 = nn.LayerNorm(self.c.fan_mid // 2)
self.linear = nn.Linear(self.c.fan_mid // 2, self.c.fan_out)
"""

class Type12(nn.Module):
    def __init__(self, config: Type12Config):
        super().__init__()
        self.c = config

        self.gcn1 = GraphConvolution(self.c.fan_in, self.c.fan_mid)
        self.ln1 = nn.LayerNorm(self.c.fan_mid)
        self.gcn2 = GraphConvolution(self.c.fan_mid, self.c.fan_mid // 2)
        self.ln2 = nn.LayerNorm(self.c.fan_mid // 2)
        self.linear = nn.Linear(self.c.fan_mid // 2, self.c.fan_out)
    
    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor]):

        x = self.gcn1(x, A_s[0])
        x = self.ln1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, self.c.dropout, training=self.training)
        x = self.gcn2(x, A_s[1])
        x = self.ln2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, self.c.dropout, training=self.training)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class Type3(nn.Module):
    def __init__(self, config: Type3Config):
        super().__init__()
        self.c = config
        self.gcn = Type12(self.c.type12_config)

    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor]):
        gcn_pred = self.gcn(x, A_s)  # already log_softmax from model
        cls_pred = F.log_softmax(self.c.cls_logit, dim=1)
    
        pred = (gcn_pred) * self.c.lmbd + cls_pred * (1 - self.c.lmbd)
   
        return pred


########################################################################################################

class Type4(nn.Module):

    def __init__(self, config: Type4Config):
        super().__init__()
        self.c = config
        self.encoder =  scGPTForAnnotation(self.c.encoder_config)
        self.gcn = Type12(self.c.type12_config) 

     
    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor], src, values, src_key_padding_mask, idx):
        encoder = self.encoder(src, values, src_key_padding_mask, batch_labels=None, cls=self.c.encoder_config.CLS)
        encoder_preds = encoder["cls_output"]
        cell_emb = encoder["cell_emb"]

        # Persist current batch's values into buffer (detached, no grad)
        with torch.no_grad():
            x[idx] = cell_emb.detach()

        # Build live input: previous batches' values from buffer (detached),
        # current batch gets live cell_emb (with grad)
        x_live = x.detach().clone()
        x_live[idx] = cell_emb  # live grad only for current batch

        gcn_pred = self.gcn(x_live, A_s)[idx]
        pred = gcn_pred * self.c.lmbd + F.log_softmax(encoder_preds, dim=1) * (1 - self.c.lmbd)
        return pred
    
    def inference(self, src, values, src_key_padding_mask):
         encoder_preds = self.encoder(src, values, src_key_padding_mask, batch_labels=None, cls=self.c.encoder_config.CLS)["cls_output"]
         pred = F.log_softmax(encoder_preds,dim=1)
         return pred





if __name__=="__main__":   

    model=torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save/dev_ms-Apr27-16-24/ms_model.pt",map_location="cpu")
    model.load_state_dict(torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save/dev_ms-Apr27-16-24/ms_model_ckpt.pt",map_location="cpu"))
    print(model)

    # Define the dimensions
    rows = 32
    cols = 700

    random_integers = torch.randint(low=1, high=5001, size=(rows, cols), dtype=torch.int64)
    src=random_integers.to("cuda")
    values= torch.rand(32,700, dtype=torch.float32).to("cuda")
    src_key_padding_mask= torch.zeros(32,700).bool().to("cuda")
    #print(src_key_padding_mask)
    model.to("cuda")

   

# 👇 Add this
    print("model dtype:", next(model.parameters()).dtype)

    
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(src, values, src_key_padding_mask)
            
            print(output["cell_emb"].size())
            print(output["cell_emb"].dtype)
  
    