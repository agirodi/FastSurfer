
# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS
import os
import glob

import requests
import torch

from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)

# Defaults
URL = "https://b2share.fz-juelich.de/api/files/a423a576-220d-47b0-9e0c-b5b32d45fc59"
VINN_AXI = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/aparc_vinn_axial_v2.0.0.pkl")
VINN_COR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/aparc_vinn_coronal_v2.0.0.pkl")
VINN_SAG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/aparc_vinn_sagittal_v2.0.0.pkl")


def create_checkpoint_dir(expr_dir, expr_num):
    """
        Create the checkpoint dir
    :param: path-like: expr_dir: [value]
    :param: path-like: expr_num: [value]
    :return: checkpoint path
    """

    checkpoint_dir = os.path.join(expr_dir, "checkpoints", str(expr_num))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def get_checkpoint(ckpt_dir, epoch):
    checkpoint_dir = os.path.join(ckpt_dir, 'Epoch_{:05d}_training_state.pkl'.format(epoch))
    return checkpoint_dir

def get_checkpoint_path(log_dir, resume_expr_num):
    """
        gets the path to the checkpoint
    :param: path-like: log_dir: path to the logs
    :param: : resume_expr_num: [value]
    :return: path to prior models
    """

    if resume_expr_num == "Default":
        return None
    checkpoint_path = os.path.join(log_dir, "checkpoints", str(resume_expr_num))
    prior_model_paths = sorted(glob.glob(os.path.join(checkpoint_path, 'Epoch_*')), key=os.path.getmtime)
    if len(prior_model_paths) == 0:
        return None
    return prior_model_paths


def load_from_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, fine_tune=False):
    """ [help]
        Loading the model from the given experiment number
    :param: path-like: checkpoint_path: path to the checkpoints
    :param: : model:
    :param: : optimizer: (Default=None)
    :param: : scheduler: (Default=None)
    :param: : fine_tune: (Default=False)
    :return: epoch number:
    """

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError:
        model.module.load_state_dict(checkpoint['model_state'])

    if not fine_tune:
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler and "scheduler_state" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint['epoch']+1, checkpoint['best_metric']


def save_checkpoint(checkpoint_dir, epoch, best_metric, num_gpus, cfg, model,  optimizer, scheduler=None, best=False):
    """ [help]
        Saving the state of training for resume or fine-tune
    :param: path-lke: checkpoint_dir: directory to the checkpoints
    :param: : epoch:
    :param: : best_metric:
    :param: int: num_gpus: number of gpus
    :param: array-like: cfg: configurations
    :param: : model:
    :param: : optimizer:
    :param: : scheduler: (Default=None)
    :param: : best: (Default=False)
    """

    save_name = f"Epoch_{epoch:05d}_training_state.pkl"
    saving_model = model.module if num_gpus > 1 else model
    checkpoint = {
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": cfg.dump()
    }

    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_dir + "/" + save_name)

    if best:
        remove_ckpt(checkpoint_dir + "/Best_training_state.pkl")
        torch.save(checkpoint, checkpoint_dir + "/Best_training_state.pkl")


def remove_ckpt(ckpt):
    try:
        os.remove(ckpt)
    except FileNotFoundError:
        pass


def download_checkpoint(download_url, checkpoint_name, checkpoint_path):
    """
        Download a checkpoint file. Raises an HTTPError if the file is not found
        or the server is not reachable.
    :param: str: download_url: URL of checkpoint hosting site
    :param: str: checkpoint_name: name of checkpoint
    :param: str: checkpoint_path: path of the file in which the checkpoint will be saved
    """

    try:
        response = requests.get(os.path.join(download_url, checkpoint_name), verify=True)
        # Raise error if file does not exist:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        LOGGER.info('Response code: {}'.format(e.response.status_code))
        response = requests.get(os.path.join(download_url, checkpoint_name), verify=False)
        response.raise_for_status()

    with open(checkpoint_path, 'wb') as f:
        f.write(response.content)


def check_and_download_ckpts(checkpoint_path, url):
    """
        Check and download a checkpoint file, if it does not exist.
    :param: str: checkpoint_path: path of the file in which the checkpoint will be saved
    :param: str: download_url: URL of checkpoint hosting site
    """

    # Download checkpoint file from url if it does not exist
    if not os.path.exists(checkpoint_path):
        ckptdir, ckptname = os.path.split(checkpoint_path)
        if not os.path.exists(ckptdir) and ckptdir:
            os.makedirs(ckptdir)
        download_checkpoint(url, ckptname, checkpoint_path)     


def get_checkpoints(axi, cor, sag, url=URL):
    """
        Check and download checkpoint files if not exist
    :param: str: url: URL of checkpoint hosting site
    :param: str: axi: path of the first file in which the checkpoint will be saved
    :param: str: cor: path of the second file in which the checkpoint will be saved
    :param: str: sag: path of the third file in which the checkpoint will be saved
    """

    check_and_download_ckpts(axi, url)
    check_and_download_ckpts(cor, url)
    check_and_download_ckpts(sag, url)
