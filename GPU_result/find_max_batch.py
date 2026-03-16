import torch
import torchvision

device = torch.device("cuda")
model = torchvision.models.regnet_y_128gf().to(device)

for bs in [8, 16, 32, 64]:
    try:
        x = torch.randn(bs, 3, 224, 224, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        print(f"batch={bs}: OK, mem={torch.cuda.memory_allocated()/1e9:.2f}GB")
        torch.cuda.empty_cache()
    except RuntimeError:
        print(f"batch={bs}: OOM")
        break
