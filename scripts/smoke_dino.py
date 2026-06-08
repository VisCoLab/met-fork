"""Smoke-test the DINOv3 integration into /met's pipeline (real siamese_network +
ContrastiveLoss). Verifies: model builds (head + LoRA), forward gives (B,1024)
descriptors, the inference resize path works for non-512 input, contrastive backward
flows grads to the trainable params (projector for head; LoRA+projector for lora) and
NOT the frozen backbone."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from code.networks.SiameseNet import siamese_network
from code.utils.losses import ContrastiveLoss

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)


def n_train(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def n_backbone_grad(m):
    # trainable params inside the DINOv3 backbone trunk (LoRA adapters live here)
    return sum(p.numel() for n, p in m.named_parameters()
               if p.requires_grad and "backbone.backbone" in n)


for lora in [False, True]:
    tag = "LoRA" if lora else "head"
    print(f"\n===== {tag} =====", flush=True)
    m = siamese_network("dinov3_vitl16", emb_proj=True, lora=lora).to(dev)
    m.train()

    # training-shape pair (2 views, batch 2, 512x512)
    x1 = torch.randn(2, 3, 512, 512, device=dev)
    x2 = torch.randn(2, 3, 512, 512, device=dev)
    d1, d2 = m(x1, x2)
    print("descriptor shapes:", tuple(d1.shape), tuple(d2.shape), flush=True)
    assert d1.shape == (2, 1024) and d2.shape == (2, 1024), (d1.shape, d2.shape)
    assert torch.allclose(d1.norm(dim=1), torch.ones(2, device=dev), atol=1e-3), "not L2-normed"

    loss = ContrastiveLoss(margin=1.8).to(dev)(d1, d2, torch.tensor([1., 0.], device=dev))
    loss.backward()
    grad_params = sum(p.numel() for p in m.parameters() if p.requires_grad and p.grad is not None)
    print(f"loss={loss.item():.4f} | trainable={n_train(m):,} | "
          f"backbone-trainable={n_backbone_grad(m):,} | got-grad={grad_params:,}", flush=True)
    assert m.backbone.projector.weight.grad is not None, "projector got no grad"
    if lora:
        assert n_backbone_grad(m) > 0, "LoRA: no trainable params in backbone"
    else:
        assert n_backbone_grad(m) == 0, "head: backbone should be frozen"

    # inference path: single non-512 image (trunk must resize internally)
    m.eval()
    with torch.no_grad():
        d = m.backbone(torch.randn(1, 3, 500, 500, device=dev))
    print("inference (500x500 -> internal resize) descriptor:", tuple(d.shape), flush=True)
    assert d.shape == (1, 1024), d.shape
    del m
    torch.cuda.empty_cache() if dev == "cuda" else None

print("\nSMOKE OK", flush=True)
