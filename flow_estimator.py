import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE = 'cuda'


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def build_raft_model(args, device):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()

    return model

@torch.no_grad()
def raft_flow(raft, map_t, map_t1, iters=20):
    """
    map_t, map_t1: [B,1,H,W] or [B,H,W] (0/1 binary or 0~1 prob)
    return: flow_up [B,2,H,W]  (dx, dy) in pixel units
    """
    if map_t.dim() == 3:
        map_t = map_t.unsqueeze(1)      # [B,1,H,W]
    if map_t1.dim() == 3:
        map_t1 = map_t1.unsqueeze(1)

    # RAFT expects 3-channel "images"
    map_t  = map_t.float().clamp(0,1) * 255.0
    map_t1 = map_t1.float().clamp(0,1) * 255.0
    img1 = map_t.repeat(1, 3, 1, 1)     # [B,3,H,W]
    img2 = map_t1.repeat(1, 3, 1, 1)

    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)

    flow_low, flow_up = raft(img1, img2, iters=iters, test_mode=True)
    flow_up = padder.unpad(flow_up)     # back to original H,W
    return flow_up
