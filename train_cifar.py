import os
from pathlib import Path
import wget
from tqdm import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
# import torchvision
from PIL import __version__ as PILLOW_VERSION
from utils import construct_rn9, get_dataloader



def train(model, loader, lr=0.4, epochs=24, momentum=0.9,
          weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0, model_id=0):

    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for ep in range(epochs):
        for it, (ims, labs) in enumerate(loader):
            ims = ims.cuda()
            labs = labs.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        if ep in [12, 15, 18, 21, 23]:
            torch.save(model.state_dict(), f'./checkpoints/sd_{model_id}_epoch_{ep}.pt')

    return model


if __name__ == "__main__":
    os.makedirs('./checkpoints', exist_ok=True)
    loader_for_training = get_dataloader(batch_size=512, split='train', shuffle=True)

    # you can modify the for loop below to train more models
    for i in tqdm(range(1), desc='Training models..'):
        model = construct_rn9().to(memory_format=torch.channels_last).cuda()
        model = train(model, loader_for_training, model_id=i)
    # For the remaining steps, we’ll assume you have N model checkpoints in ./checkpoints:

    import torch
    from pathlib import Path

    ckpt_files = list(Path('./checkpoints').rglob('*.pt'))
    ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]