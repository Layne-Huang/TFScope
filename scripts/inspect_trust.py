import os,sys,json,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,"src")
os.environ.setdefault("TORCH_HOME","/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from torch.utils.data import DataLoader
CK="/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v14_icpcc/ckpt_best.pt"
TARGET="1le8_B_MATALPHA2.MA0328.2.txt"
cfg=TFScopeConfig()
for k,v in json.load(open(os.path.join(os.path.dirname(CK),"config.json"))).items():
    if hasattr(cfg,k):
        try:setattr(cfg,k,type(getattr(cfg,k))(v))
        except:pass
cfg.retrieval_index_path="data/processed/tf_nn_index_lgo_deeppbs.json"
dev="cuda" if torch.cuda.is_available() else "cpu"
m=TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CK,map_location=dev,weights_only=False)["model"],strict=False)
ds=TFDataset(cfg,"data/processed/tf_pwm_deeppbs_only.parquet","data/processed/splits/deeppbs_only/benchmark_no_val.json",split="test",max_seq_len=1024)
i=ds.filenames.index(TARGET)
batch=collate_variable_length([ds[i]])
batch={k:v.to(dev,dtype=torch.float32 if v.is_floating_point() else torch.long) for k,v in batch.items()}
with torch.no_grad():
    gl,pl,aux=m(batch["sequence_tokens"],batch["dbd_mask"],batch["family_id"],
                retrieved_pwms=batch.get("retrieved_pwms"),retrieved_masks=batch.get("retrieved_masks"),retrieved_sims=batch.get("retrieved_sims"))
tl=aux["trust_logits"][0].cpu().numpy(); trust=1/(1+np.exp(-tl))
sims=batch["retrieved_sims"][0].cpu().numpy()
h=m.pwm_head
rb=float(h.retrieval_beta.detach().cpu()); cs=float(h.conf_scale.detach().cpu()); ct=float(h.conf_thresh.detach().cpu())
gate_in=trust.max()
beta_gated=rb*(1/(1+np.exp(-cs*(gate_in-ct))))
print(f"TF: {TARGET}")
print(f"retrieved cos sims: {sims.round(3)}")
print(f"trust logits      : {tl.round(3)}")
print(f"trust scores (sig): {trust.round(3)}")
print(f"\nhead params: retrieval_beta={rb:.3f}  conf_scale={cs:.3f}  conf_thresh={ct:.3f}")
print(f"gate_input (max trust) = {gate_in:.3f}")
print(f"=> beta_gated = {beta_gated:.4f}")
print(f"\nInterpretation: final = de_novo_logits + {beta_gated:.3f} * combined_log_prior")
