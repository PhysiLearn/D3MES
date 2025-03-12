import torch
import numpy as np
import pandas as pd
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from models_random_drugs import DiT_models
import argparse
import os

def find_model(model_path):
    """
    Loads a model from a local path.
    """
    assert os.path.isfile(model_path), f'Could not find DiT checkpoint at {model_path}'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    # Load a custom DiT checkpoint:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Labels to condition the model with:
    class_labels = [0]*10000

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 3, latent_size, 3, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # 生成数据
    generated_data = samples.cpu().numpy()
    output_npz_path = "drugsgen.npz"
    
    # 保存为npz文件
    np.savez(output_npz_path, data=generated_data)
    print(f"Generated data saved to {output_npz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-ZZJ")
    parser.add_argument("--image-size", type=int, default=75)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="/path/to/pt.pt", help="Path to a DiT checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)

