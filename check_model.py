import os
import torch

p = 'models/stage1_rice.pth'

if os.path.exists(p):
    ckpt = torch.load(p, map_location='cpu')
    print('Model loaded successfully')
    print('Val acc :', round(ckpt["val_acc"] * 100, 2), '%')
    print('Epoch   :', ckpt["epoch"] + 1)
    print('Classes :', ckpt["classes"])
    print('Num cats:', ckpt["num_categories"])
    size_mb = os.path.getsize(p) / (1024 * 1024)
    print('Size    :', round(size_mb, 1), 'MB')
else:
    print('NOT FOUND at:', os.path.abspath(p))
    print('Copy stage1_rice.pth into the models/ folder first')