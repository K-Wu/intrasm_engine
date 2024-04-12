import math
import time

import torch
import torch.nn as nn
import transformers


# Adapted from github.com/IST-DASLab/sparsegpt/sparsegpt.py
def fasterprune(
    self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=0.01
):
    W = self.layer.weight.data.clone()
    if isinstance(self.layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    tick = time.time()

    H = self.H
    del self.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    Losses = torch.zeros(self.rows, device=self.dev)

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.columns, device=self.dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    mask = None

    for i1 in range(0, self.columns, blocksize):
        i2 = min(i1 + blocksize, self.columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if prunen == 0:
            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][
                    int(tmp.numel() * sparsity)
                ]
                mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1) == 1

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if prunen != 0 and i % prunem == 0:
                tmp = (
                    W1[:, i : (i + prunem)] ** 2
                    / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1)))
                    ** 2
                )
                mask1.scatter_(
                    1,
                    i + torch.topk(tmp, prunen, dim=1, largest=False)[1],
                    True,
                )

            q = w.clone()
            q[mask1[:, i]] = 0

            # For simplicity, quantization is removed during adaptation

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        Losses += torch.sum(Losses1, 1) / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize()
    print("time %.2f" % (time.time() - tick))
    print("error", torch.sum(Losses).item())

    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
        self.layer.weight.data.dtype
    )
