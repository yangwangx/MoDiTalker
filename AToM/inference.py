from functools import partial
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import os
import torch
import numpy as np
import importlib
import random

from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import MotionDecoder

import torch.nn.functional as F
import pdb
import cv2

import glob
import argparse

from data_util.face3d_helper import Face3DHelper

# =========================================
HORIZON = 156       # sequence length
# =========================================
feature_dim = 1024  # audio
repr_dim = 204      # identity (frontalized 3d landmarks)
device = "cuda"
seed = 2021
deterministic = True

def prepare_models(args):
    # Create AToM model
    model = MotionDecoder(
        nfeats=repr_dim,
        seq_len=HORIZON,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=feature_dim,
        activation=F.gelu,
    )
    print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

    # Load checkpoint weight
    checkpoint = torch.load(args.checkpoint, map_location=device)
    EMA = False
    model.load_state_dict(maybe_wrap(checkpoint["ema_state_dict" if EMA else "model_state_dict"], 1))

    # Create diffusion pipeline
    diffusion = GaussianDiffusion(
        model,
        HORIZON,
        repr_dim,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=2,
    )
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    face3d_helper = Face3DHelper(args.face3d_helper)
    
    return model, diffusion, face3d_helper


def save_lm_img(lm3D, out_path, WH=256, flip=True):
    if lm3D.shape[-1] == 3:
        # shape [..., 3], range [-1, 1]
        lm3d = (lm3D * WH / 2 + WH / 2).astype(int)
        lm2d = lm3d[:, :2]
    else:
        lm2d = lm3D.astype(int)
    # img: white background, draw landmarks as black dots
    img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
    for i in range(len(lm2d)):
        x, y = lm2d[i]
        img = cv2.circle(img, center=(x, y), radius=3, color=(0, 0, 0), thickness=-1)

    if flip:
        img = cv2.flip(img, 0)

    cv2.imwrite(out_path, img)


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


def load_idlist(path):
    with open(path, "r") as f:
        lines = f.readlines()
        id_list = [line.replace("\n", "").replace(".mp4", "").strip() for line in lines]
    return id_list


def main(args):
    model, diffusion, face3d_helper = prepare_models(args)
    if args.id_list is None : 
        id_list = os.listdir(os.path.join(args.data_root, "frames"))
    else:
        id_list = load_idlist(args.id_list)
    
    for name in tqdm(id_list):
        for it in range(HORIZON // 156):
            STR = it * HORIZON
            END = (it + 1) * HORIZON

            # Load `id_cond`, identity, frontalized face landmarks
            if it == 0:
                id_cond = np.load(os.path.join(args.data_root, "keypoints/face-centric/unposed/{name}/00000.npy"))
                # id_cond = np.load(f"../data/inference/init_kpt/{name}.npy")
                id_cond = torch.from_numpy(id_cond)
                id_cond = id_cond[:, 0:1, :].to(device)
            else : #TODO
                id_cond = np.load(f"")             
            if id_cond.shape[-1] == 3:
                id_cond = id_cond.unsqueeze(0)
                id_cond = id_cond.view(1, 1, -1)
            id_cond = torch.cat([id_cond for _ in range(HORIZON)], dim=1) 

            # Load `audio_cond`, audio features
            hubert_name = args.hubert_path
            audio_cond = np.load(hubert_name) 
            audio_cond = torch.from_numpy(audio_cond)
            audio_cond = audio_cond.unsqueeze(0)  
            audio_cond = audio_cond[:, STR : (it + 2 * HORIZON), :].to(device) 


            landmark_shape = [1, HORIZON, repr_dim]
            landmark_noise = torch.rand(1, HORIZON, repr_dim, device=device)
            pos = torch.zeros(1, HORIZON, 3, device=device)  # why 3?

            with torch.no_grad():
                _epoch = 0
                _render_out = ""
                atom_out = diffusion.render_sample(
                    face3d_helper,
                    landmark_shape,
                    id_cond,
                    landmark_noise,
                    pos,
                    landmark_noise,
                    audio_cond,
                    _epoch
                    _render_out
                    name=name,
                    sound=True,
                )

            # atom_out is the predicted motion from the id_cond
            atom_out = atom_out[0]
            atom_out = atom_out.view(HORIZON, -1).detach().cpu()
            atom_out += id_cond.view(HORIZON, -1).detach().cpu()
            atom_out = atom_out / 10 + face3d_helper.key_mean_shape.squeeze().reshape([1, -1]).cpu().numpy()
            atom_out = atom_out.view(HORIZON, 68, 3)
            atom_out = atom_out.cpu().numpy()

            frontalized_npy_save_dir = os.path.join(args.save_dir, "frontalized_npy", f"{name}")
            os.makedirs(frontalized_npy_save_dir, exist_ok=True)
            np.save(f"{frontalized_npy_save_dir}/atom_{str(it)}.npy", atom_out)
            # ------------------------- visualization------------------------------------------ #
            frontalized_png_save_dir = os.path.join(args.save_dir, "frontalized_png", f"{name}")
            os.makedirs(frontalized_png_save_dir, exist_ok=True)
            atom_out = (atom_out * 256 / 2 + 256 / 2).astype(int)
            for i_img in range(156):
                vis_atom_out = atom_out[i_img, :, :2]
                img = np.ones([256, 256, 3], dtype=np.uint8) * 255
                for i in range(68):
                    x, y = vis_atom_out[i]
                    img = cv2.circle(img, center=(x, y), radius=3, color=(0, 0, 0), thickness=-1)
                img = cv2.flip(img, 0)
                out_name = f"{frontalized_png_save_dir}/{str(it*HORIZON +i_img).zfill(3)}.png"
                cv2.imwrite(out_name, img)
            # --------------------------------------------------------------------------------- #

            print(f"Done")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="")
    args.add_argument("--data_root", type=str,
                      default="../data/train/HDTF", help="data location of reference images")
    args.add_argument("--hubert_path", type=str,
                      default="../data/inference/hubert/16000/LetItGo.npy", help="path to the hubert extracted")
    args.add_argument("--face3d_helper", type=str,
                      default="../data/data_utils/deep_3drecon/BFM", help="path to the BFM folder")
    args.add_argument("--id_list", type=str,
                      default=None, help="if id_list is None, then the whole id in the data_root will be included")
    args.add_argument("--device", type=str,
                      default="cuda:5")
    args.add_argument("--checkpoint", type=str,
                      help="path to the checkpoint of AToM")
    args.add_argument("--save_dir", type=str,
                      default="results/frontalized1", help="path to the directory to save frontalized landmarks")
    args = args.parse_args()
    main(args)
