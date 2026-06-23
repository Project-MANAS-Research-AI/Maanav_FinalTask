import torch
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.eval()
model.to(device)

print("Model loaded.")
print("Number of transformer blocks:", len(model.blocks))

class AttentionGrabber:
    """Patches every block's attention module so it saves its attention
    matrix to a list, instead of just using it and discarding it."""

    def __init__(self, model):
        self.attentions = []
        for block in model.blocks:
            self._patch(block.attn)

    def _patch(self, attn_module):
        num_heads = attn_module.num_heads
        scale = attn_module.scale

        def forward(x, attn_mask=None, is_causal=False):
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(
                B, N, 3, num_heads, C // num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

            self.attentions.append(attn.detach().mean(dim=1).cpu())

            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            out = attn_module.proj(out)
            return out

        attn_module.forward = forward

    def reset(self):
        self.attentions = []


grabber = AttentionGrabber(model)
print("Attention grabber attached.")

from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

img = Image.open("images/dog.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0).to(device)

grabber.reset()
with torch.no_grad():
    logits = model(x)

print("Input shape:", x.shape)
print("Number of attention matrices captured:", len(grabber.attentions))
print("Shape of one attention matrix:", grabber.attentions[0].shape)
print("Predicted class index:", logits.argmax(dim=-1).item())
def attention_rollout(attentions):
    """attentions: list of 12 tensors, each shape (1, 197, 197).
    Returns: one (1, 197, 197) tensor — the rolled-out attention."""

    B, N, _ = attentions[0].shape
    result = torch.eye(N).unsqueeze(0).repeat(B, 1, 1) 

    for attn in attentions:
        attn = attn + torch.eye(N).unsqueeze(0)          
        attn = attn / attn.sum(dim=-1, keepdim=True)     
        result = torch.bmm(attn, result)                  

    return result


rollout = attention_rollout(grabber.attentions)
cls_attention = rollout[0, 0, 1:]  

print("Rollout shape:", rollout.shape)
print("CLS attention to patches shape:", cls_attention.shape)
print("Min/max attention value:", cls_attention.min().item(), cls_attention.max().item())

import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

grid = cls_attention.reshape(14, 14).numpy()

grid = (grid - grid.min()) / (grid.max() - grid.min())

heatmap = torch.tensor(grid).unsqueeze(0).unsqueeze(0) 
heatmap = F.interpolate(heatmap, size=(224, 224), mode="bilinear")
heatmap = heatmap.squeeze().numpy()

img_resized = img.resize((224, 224))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_resized)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(heatmap, cmap="jet")
axes[1].set_title("Attention heatmap")
axes[1].axis("off")

axes[2].imshow(img_resized)
axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/rollout_dog.png", dpi=150)
print("Saved outputs/rollout_dog.png")

import os
from urllib.request import urlopen

labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urlopen(labels_url).read().decode("utf-8").splitlines()


def run_and_visualize(image_path, out_path):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    grabber.reset()
    with torch.no_grad():
        logits = model(x)
    pred_idx = logits.argmax(dim=-1).item()
    pred_label = imagenet_classes[pred_idx]

    rollout = attention_rollout(grabber.attentions)
    cls_attention = rollout[0, 0, 1:]

    grid = cls_attention.reshape(14, 14).numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min())

    heatmap = torch.tensor(grid).unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(224, 224), mode="bilinear")
    heatmap = heatmap.squeeze().numpy()

    img_resized = img.resize((224, 224))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_resized); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Attention heatmap"); axes[1].axis("off")
    axes[2].imshow(img_resized)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay"); axes[2].axis("off")

    fig.suptitle(f"{os.path.basename(image_path)} -> {pred_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"{os.path.basename(image_path):15s} pred='{pred_label}'  saved -> {out_path}")


for fname in os.listdir("images"):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        in_path = os.path.join("images", fname)
        out_path = os.path.join("outputs", f"rollout_{os.path.splitext(fname)[0]}.png")
        run_and_visualize(in_path, out_path)