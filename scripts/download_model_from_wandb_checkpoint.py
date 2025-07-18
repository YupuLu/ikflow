from typing import Dict
import argparse
import os
from time import time
import pickle

import wandb
import torch

from ikflow.utils import get_wandb_project, safe_mkdir
from ikflow.config import MODELS_DIR

def format_state_dict(state_dict: Dict) -> Dict:
    """The `state_dict` saved in checkpoints will have keys with the form:
        "nn_model.module_list.0.M", "nn_model.module_list.0.M_inv", ...

    This function updates them to the expected format below. (note the 'nn_model' prefix is removed)
        "module_list.0.M", "module_list.0.M_inv", ...
    """
    bad_prefix = "nn_model."
    len_prefix = len(bad_prefix)
    updated = {}
    for k, v in state_dict.items():
        # Check that the original state dict is malformatted first
        assert k[0:len_prefix] == bad_prefix
        k_new = k[len_prefix:]
        updated[k_new] = v
    return updated


"""
_____________
Example usage

python scripts/download_model_from_wandb_checkpoint.py --wandb_run_id=2uidt835
python scripts/download_model_from_wandb_checkpoint.py --wandb_run_id=2uidt835 --model_id=v8
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Download ikflow model from Weights and Biases")

    # Note: WandB saves artifacts by the run ID (i.e. '34c2gimi') not the run name ('dashing-forest-33'). This is
    # slightly annoying because you need to click on a run to get its ID.
    parser.add_argument("--wandb_run_id", type=str, help="The run ID of the wandb run to load. Example: '34c2gimi'")
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--model_id", type=str, default="best_k", help="The model ID of the wandb run to load. Example: 'best_k', 'v16'")
    parser.add_argument("--local_artifact", action="store_true",
                        help="If set, uses the local artifact cache instead of downloading from the cloud.")
    parser.add_argument("--ckpt_filepath", type=str, default=None,
                        help="The path to the checkpoint file to load. Should be provided if `--local_artifact` is set.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="The name of the run to load. Should be provided if `--local_artifact` is set.")
    parser.add_argument("--robot_name", type=str, default=None,
                        help="The name of the robot to load. Should be provided if `--local_artifact` is set.")
    args = parser.parse_args()

    if not args.local_artifact:
        print("Downloading model from WandB...")
        wandb_entity, wandb_project = get_wandb_project()
        t0 = time()
        api = wandb.Api()
        artifact = api.artifact(f"{wandb_entity}/{wandb_project}/model-{args.wandb_run_id}:{args.model_id}")
        download_dir = artifact.download()
        print(f"Downloaded artifact in {round(time()- t0, 2)}s")

        t0 = time()
        run = api.run(f"/{wandb_entity}/{wandb_project}/runs/{args.wandb_run_id}")
        print(f"Downloaded run data in {round(time()- t0, 2)}s")

        run_name = run.name
        robot_name = run.config["robot"]
        ckpt_filepath = os.path.join(download_dir, "model.ckpt")
    else:
        if args.ckpt_filepath is None or args.run_name is None or args.robot_name is None:
            raise ValueError("If `--local_artifact` is set, `--ckpt_filepath`, `--run_name`, and `--robot_name` must be set. The script will use the local artifact cache.")
        print("Using local artifact cache...")
        run_name = args.run_name
        robot_name = args.robot_name
        ckpt_filepath = args.ckpt_filepath
    checkpoint = torch.load(ckpt_filepath, map_location=lambda storage, loc: storage)
    state_dict = format_state_dict(checkpoint["state_dict"])
    global_step = str(checkpoint["global_step"] / 1e6) + "M"

    # Save model's state_dict
    safe_mkdir(MODELS_DIR)
    model_state_dict_filepath = os.path.join(MODELS_DIR, f"{robot_name}__{run_name}__global_step_{global_step}.pkl")
    if os.path.exists(model_state_dict_filepath):
        print(f"Model state dict already exists at {model_state_dict_filepath}. Overwriting it.")
    print(f"Saving model state dict to {model_state_dict_filepath}")
    with open(model_state_dict_filepath, "wb") as f:
        pickle.dump(state_dict, f)
