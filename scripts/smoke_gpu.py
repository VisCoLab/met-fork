"""GPU smoke test for the Met pipeline: H100 torch forward + CPU-faiss kNN/mining.
Run on a GPU node:  srun ... .venv/bin/python -u data/smoke_gpu.py
"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
os.environ.setdefault("TORCH_HOME", os.path.join(REPO, "data", "torch_home"))
sys.path.insert(0, REPO)                                    # so `import code.*` works

import torch, numpy as np, faiss

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "NO CUDA on this node"
print("GPU:", torch.cuda.get_device_name(0), "| faiss.get_num_gpus():", faiss.get_num_gpus(), flush=True)

# 1) real SWSL R18 (+FC) forward on GPU at training resolution (from cached hub)
from code.networks.SiameseNet import siamese_network
m = siamese_network("r18_sw-sup", pooling="gem", pretrained=False, emb_proj=True).cuda().eval()
with torch.no_grad():
    desc = m.backbone(torch.randn(4, 3, 500, 500, device="cuda")).cpu().numpy().astype("float32")
print("SWSL R18 GPU forward OK -> descriptors", desc.shape, "| L2 norm:", round(float(np.linalg.norm(desc[0])), 3), flush=True)

# 2) patched KNN_Classifier (CPU faiss) fit + predict
from code.classifiers.knn_classifier import KNN_Classifier
clf = KNN_Classifier(K=1, t=1)
clf.fit(desc, np.array([0, 1, 2, 3]))
preds, confs = clf.predict(desc)
print("KNN_Classifier (CPU faiss) OK -> preds", list(map(int, preds)), flush=True)

# 3) patched mine_negatives (CPU faiss) over a small synthetic set
from code.utils.train_utils import mine_negatives
N = 200
dd = np.random.rand(N, 512).astype("float32"); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
negs = mine_negatives([f"p{i}" for i in range(N)], ".", dd, np.array([i % 10 for i in range(N)]))
print("mine_negatives (CPU faiss) OK ->", len(negs), "negatives mined", flush=True)
print("SMOKE TEST PASSED", flush=True)
