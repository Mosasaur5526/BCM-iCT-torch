"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
sys.path.append('../BCM')

from PIL import Image

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from cm.image_datasets import load_data


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=os.path.join(args.save_dir, args.exp_name))

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    if args.eval_mse:
        logger.log("loading test data for reconstruction MSE evaluation...")
        # test data
        data = load_data(
            data_dir=args.test_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            val=True
        )
    else:
        data = None

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
        model_type=args.model_type
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    model_path = args.model_path
    model_path = model_path.split('/')[-1][:-3]

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_images2 = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    while len(all_images) * args.batch_size < args.num_samples:
        if not args.eval_mse:
            # then perform generation
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            # now, sample is the initial Gaussian noise at T=80.0
            sample = th.randn((args.batch_size, 3, args.image_size, args.image_size), device=dist_util.dev()) * 80.0
            multiplier = th.ones(sample.shape[0], dtype=sample.dtype, device=sample.device)

            # ----------------- one-step generation -----------------
            # 80.0 -> 0.002
            _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 80.0, sigmas_end=multiplier * 0.002, y=classes)
            # ----------------------------------------------------------

            # -------------- two-step ancestral sampling ------------
            # 80.0 -> 2.4 -> 0.002
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 80.0, sigmas_end=multiplier * 2.4, y=classes)
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 2.4, sigmas_end=multiplier * 0.002, y=classes)
            # ----------------------------------------------------------

            # -------------- three-step zigzag sampling ------------
            # 80.0 -> 0.002 -> 0.1 (manually added noise) -> 1.2 (amplified by BCM) -> 0.002
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 80.0, sigmas_end=multiplier * 0.002, y=classes)
            # sample += th.randn_like(sample, device=sample.device) * 0.1
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 0.1, sigmas_end=multiplier * 1.2, y=classes)
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 1.2, sigmas_end=multiplier * 0.002, y=classes)
            # ----------------------------------------------------------

            # -------------- four-step mixed sampling --------------
            # 80.0 -> 3.0 -> 0.002
            # -> 0.12 (manually added noise) -> 0.4 (amplified by BCM) -> 0.002
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 80.0, sigmas_end=multiplier * 3.0, y=classes)
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 3.0, sigmas_end=multiplier * 0.002, y=classes)
            # sample += th.randn_like(sample, device=sample.device) * 0.12
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 0.12, sigmas_end=multiplier * 0.4, y=classes)
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 0.4, sigmas_end=multiplier * 0.002, y=classes)
            # ----------------------------------------------------------

        else:
            # to perform inversion and evaluate reconstrution MSE, first load samples from val set
            batch, classes = next(data)
            batch = batch.to(device=dist_util.dev())
            classes = classes['y'].to(device=dist_util.dev())
            sample = batch

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        if args.reconstruct or args.eval_mse:
            # add a small initial noise
            multiplier = th.ones(sample.shape[0], dtype=sample.dtype, device=sample.device)
            sample = sample + th.randn_like(sample, device=sample.device) * 0.07

            # ------------------- one-step inversion -----------------
            # _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 0.07, sigmas_end=multiplier * 80.0, y=classes)
            # ----------------------------------------------------------

            # ------------------- two-step inversion -----------------
            _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 0.07, sigmas_end=multiplier * 15.0, y=classes)
            _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 15.0, sigmas_end=multiplier * 80.0, y=classes)
            # ----------------------------------------------------------

            # --------------------- reconstrution --------------------
            _, sample = diffusion.denoise(model, sample, sigmas=multiplier * 80.0, sigmas_end=multiplier * 0.002, y=classes)
            # ----------------------------------------------------------

            sample = sample.permute(0, 2, 3, 1)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images2.extend([sample.cpu().numpy() for sample in gathered_samples])

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.reconstruct or args.eval_mse:
        arr2 = np.concatenate(all_images2, axis=0)
        arr2 = arr2[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        name = 'samples_original' if args.reconstruct or args.eval_mse else 'samples'
        name2 = 'Original samples' if args.reconstruct or args.eval_mse else 'Samples'
        out_path = os.path.join(logger.get_dir(), f"{model_path}_{name}_{shape_str}.npz")
        logger.log(f"{name2} saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        if args.reconstruct or args.eval_mse:
            out_path2 = os.path.join(logger.get_dir(), f"{model_path}_samples_reconstructed.npz")
            np.savez(out_path2, arr2)
            logger.log(f"Reconstructed images saving to {out_path2}")

            mse = (((arr / 255. - arr2 / 255.).reshape(-1)) ** 2).mean()
            logger.log(f"Reconstruction per pixel MSE: {mse}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        save_dir='./checkpoints',
        exp_name='ict',
        seed=42,
        ts="",
        reconstruct=False,
        test_data_dir='/mnt/petrelfs/share/images/val',
        eval_mse=False,
        model_type='tsinghua'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
