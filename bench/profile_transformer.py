"""Quick profile of one transformer call to find hot spots."""
import os, sys, time, torch
os.environ.setdefault("PYTHONUSERBASE", "/datahdd/agent/retrochimera/.userbase")
sys.path.insert(0, "/datahdd/agent/retrochimera/RetroChimera")
import json
from syntheseus import Molecule

CKPT_DIR = "/datahdd/model_paper_results/pistachio/checkpoints/ensemble"
mols = []
with open("/data/projects/reaction_prediction/data/preprocessed/pistachio_2023Q2_v2/test.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 16: break
        mols.append(Molecule(json.loads(line)["products"][0]["smiles"]))

from retrochimera.inference.smiles_transformer import SmilesTransformerModel
model = SmilesTransformerModel(model_dir=os.path.join(CKPT_DIR, "smiles_transformer"), device="cuda:0")

# warmup
for _ in range(2):
    _ = model(mols[:8], num_results=10)
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(2):
        _ = model(mols[:8], num_results=10)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
print("---CPU---")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
