"""Utility helpers for running AttnGAN training and inference inside Jupyter notebooks.

This module wraps the existing training scripts so they can be reused from
interactive environments.  Functions here intentionally mirror the behaviour of
``main.py`` and ``main_poem_eval.py`` while avoiding command-line argument
parsing so notebooks can call them directly.
"""
from __future__ import annotations

import datetime
import os
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import dateutil.tz
import numpy as np
import torch
import torchvision.transforms as transforms
from nltk.tokenize import RegexpTokenizer

from miscc.config import cfg, cfg_from_file
from trainer import condGANTrainer
from datasets import ChiTextDataset, TextDataset


def _set_random_seeds(seed: Optional[int]) -> int:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Manual seed to use.  When ``None`` a random seed is drawn.

    Returns:
        The seed that was ultimately used so notebooks can record it.
    """
    if seed is None:
        seed = random.randint(1, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(seed)
    return seed


def _resolve_dataset_class() -> type:
    """Return the dataset class that matches the active configuration."""
    dataset_name = cfg.DATASET_NAME.lower()
    if "zikai" in dataset_name or "poem" in dataset_name:
        return ChiTextDataset
    return TextDataset


def _build_dataloader(split: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """Create the dataset and data loader that mirror the CLI entrypoints."""
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose(
        [
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
        ]
    )

    dataset_cls = _resolve_dataset_class()
    dataset = dataset_cls(
        cfg.DATA_DIR,
        split,
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True,
        shuffle=(split == "train"),
        num_workers=int(cfg.WORKERS),
    )
    return dataset, dataloader


def prepare_trainer(
    cfg_path: str,
    data_dir: Optional[str] = None,
    gpu_id: int = -1,
    manual_seed: Optional[int] = None,
    split: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[condGANTrainer, torch.utils.data.Dataset, torch.utils.data.DataLoader, str, int]:
    """Load configuration and return a ready-to-run ``condGANTrainer`` instance.

    Args:
        cfg_path: Path to the YAML configuration file.
        data_dir: Optional override for the dataset directory.
        gpu_id: GPU index to use.  ``-1`` disables CUDA and runs on CPU.
        manual_seed: Optional manual seed.  When omitted a random seed is used.
        split: Dataset split to load.  Defaults to ``'train'`` when training and
            ``'test'`` otherwise.
        output_dir: Optional override for where checkpoints/images should be
            written.  When omitted a timestamped directory is created under
            ``../output`` relative to the AttnGAN project root.

    Returns:
        A tuple ``(trainer, dataset, dataloader, output_dir, used_seed)``.
    """
    cfg_from_file(cfg_path)

    if gpu_id != -1:
        cfg.GPU_ID = gpu_id
    else:
        cfg.CUDA = False

    if data_dir:
        cfg.DATA_DIR = data_dir

    if split is None:
        split = "train" if cfg.TRAIN.FLAG else "test"

    used_seed = _set_random_seeds(manual_seed if cfg.TRAIN.FLAG else 100)

    if output_dir is None:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        output_dir = os.path.join(
            "..", "output", f"{cfg.DATASET_NAME}_{cfg.CONFIG_NAME}_{timestamp}"
        )

    dataset, dataloader = _build_dataloader(split)
    trainer = condGANTrainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, dataset)
    return trainer, dataset, dataloader, output_dir, used_seed


def run_training_loop(trainer: condGANTrainer) -> None:
    """Kick off AttnGAN training using the provided ``condGANTrainer`` instance."""
    trainer.train()


def build_custom_example(prompts: Sequence[str], wordtoix: Mapping[str, int]) -> Dict[str, List[np.ndarray]]:
    """Convert free-form prompts into the ``data_dic`` structure expected by ``gen_example``.

    Args:
        prompts: Iterable of textual prompts to visualise.
        wordtoix: Token-to-index mapping from the dataset.

    Returns:
        Dictionary mapping synthetic file keys to ``(captions, lengths, indices)``.
    """
    tokenizer = RegexpTokenizer(r"\w+")
    data_dic: Dict[str, List[np.ndarray]] = {}
    for idx, prompt in enumerate(prompts):
        cleaned = prompt.replace("\ufffd\ufffd", " ").lower()
        tokens = [token.encode("ascii", "ignore").decode("ascii") for token in tokenizer.tokenize(cleaned)]
        indices = [wordtoix[token] for token in tokens if token in wordtoix]
        if not indices:
            raise ValueError(
                f"Prompt '{prompt}' does not contain any tokens present in the vocabulary."
            )
        cap_array = np.zeros((1, len(indices)), dtype="int64")
        cap_array[0, : len(indices)] = indices
        cap_lens = np.array([len(indices)], dtype="int64")
        sorted_indices = np.array([0], dtype="int64")
        data_dic[f"prompt_{idx:02d}"] = [cap_array, cap_lens, sorted_indices]
    return data_dic


def generate_from_prompts(
    trainer: condGANTrainer,
    prompts: Sequence[str],
    wordtoix: Mapping[str, int],
) -> Dict[str, List[Path]]:
    """Generate images for custom prompts using a pre-trained generator.

    ``cfg.TRAIN.NET_G`` and ``cfg.TRAIN.NET_E`` must point to trained checkpoints
    before calling this helper.

    Args:
        trainer: The trainer returned by :func:`prepare_trainer`.
        prompts: Collection of prompt strings to render.
        wordtoix: Dataset vocabulary dictionary.

    Returns:
        Mapping from synthetic prompt keys to the list of generated image paths.
    """
    if not cfg.TRAIN.NET_G:
        raise ValueError("cfg.TRAIN.NET_G must reference a trained generator checkpoint.")
    if not cfg.TRAIN.NET_E:
        raise ValueError("cfg.TRAIN.NET_E must reference a trained text encoder checkpoint.")

    data_dic = build_custom_example(prompts, wordtoix)
    trainer.gen_example(data_dic)

    base_dir = Path(cfg.TRAIN.NET_G).with_suffix("")
    results: Dict[str, List[Path]] = {}
    for key in data_dic:
        prompt_dir = base_dir / key
        if not prompt_dir.exists():
            continue
        generated = sorted(prompt_dir.glob("*_g*.png"))
        results[key] = generated
    return results


__all__ = [
    "prepare_trainer",
    "run_training_loop",
    "generate_from_prompts",
    "build_custom_example",
]
