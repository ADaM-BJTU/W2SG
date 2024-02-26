import json
import os
import random
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,4'

from typing import Dict, List, Optional, Sequence, Union

import fire
import numpy as np
import torch
# import tiktoken
import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import tokenize_dataset
from datasets import load_dataset,load_from_disk
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss, weight_xent_loss, mse_loss_fn
from weak_to_strong.train import ModelConfig, train_and_save_model


MODEL_CONFIGS = [
    ModelConfig(
        # name="gpt2",
        name = "gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
        custom_kwargs={
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
        custom_kwargs={
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
        custom_kwargs={
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        custom_kwargs={
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="qwen-1.8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="qwen-14B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]


MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}

loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
    "weight_xent":weight_xent_loss(),
    "mse": mse_loss_fn(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

E = int(sys.argv[1])
# E = 2
seed_torch(E)
def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    train1_name: str = "./sciq/gradient_boost/train1_10000_{}/".format(E),
    train2_name: str = "./sciq/train2/",
    test_name: str = "./sciq/test",
    transfer_loss: Union[str, Sequence[str]] = "xent,logconf",
    n_docs: int = 10000,
    n_test_docs: int = 200,
    weak_model_size: str = "gpt2",
    weak_lr: Optional[float] = None,
    strong_model_size: str = "qwen-7B",
    strong_lr: Optional[float] = None,
    # Defaults to strong_lr
    transfer_lr: Optional[float] = None,
    # Optims default to default_optimizer in the model definitions
    weak_optim: Optional[str] = None,
    strong_optim: Optional[str] = None,
    transfer_optim: Optional[str] = None,
    gt_epochs: int = 3,
    # defaults to gt_epochs
    transfer_epochs: Optional[int] = None,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "./results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    log_prefix: str = "",
    # Set to an absurdly high value so we don't do intermediate evals by default.
    eval_every: int = 100000000,
):
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1
 
    if isinstance(transfer_loss, str):
        transfer_losses = transfer_loss.split(",")
    else:
        transfer_losses = transfer_loss
    del transfer_loss
    for tloss in transfer_losses:
        assert tloss in VALID_LOSSES, f"Unknown loss {tloss} not in {VALID_LOSSES}"
    assert (
        weak_model_size in MODELS_DICT
    ), f"Unknown model size {weak_model_size} not in {MODELS_DICT}"
    weak_model_config = MODELS_DICT[weak_model_size]
    assert (
        strong_model_size in MODELS_DICT
    ), f"Unknown model size {strong_model_size} not in {MODELS_DICT}"
    strong_model_config = MODELS_DICT[strong_model_size]

    if weak_lr is None:
        assert batch_size == 32
        weak_lr = weak_model_config.default_lr
    if strong_lr is None:
        assert batch_size == 32
        strong_lr = strong_model_config.default_lr
    if transfer_lr is None:
        transfer_lr = strong_lr
    if transfer_epochs is None:
        transfer_epochs = gt_epochs

    if weak_optim is None:
        weak_optim = weak_model_config.default_optimizer
    if strong_optim is None:
        strong_optim = strong_model_config.default_optimizer
    if transfer_optim is None:
        transfer_optim = strong_optim

    weak_eval_batch_size = weak_model_config.eval_batch_size
    strong_eval_batch_size = strong_model_config.eval_batch_size

    # Load dataset

    train1_ds = load_from_disk(train1_name)
    train2_ds = load_from_disk(train2_name)
    test_ds = load_from_disk(test_name)

    print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

    def train_model(
        model_config: ModelConfig,
        train_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        *,
        loss_type: str,
        label: str,
        subpath,
        lr,
        eval_batch_size,
        epochs=1,
        inference_ds: Optional[torch.utils.data.Dataset] = None,
        linear_probe: bool = False,
        optimizer_name: str = "adam",
    ):
        save_path = os.path.join(results_folder, subpath)
        linprobe_str = "_linprobe" if linear_probe else ""
        logger.configure(
            name="{log_prefix}{label}_{base_model_name}_{train1_name}_{loss_type}_{optimizer_name}_{lr}_{lr_schedule}{linprobe_str}_{datetime_now}",
            label=label,
            ds_name=train1_name,
            truncation_max_len=n_docs or "none",
            loss_type=loss_type,
            lr=lr,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            minibatch_size_per_device=minibatch_size_per_device,
            save_path=save_path,
            base_model_name=model_config.name,
            epochs=epochs,
            linprobe_str=linprobe_str,
            lr_schedule=lr_schedule,
            log_prefix=log_prefix,
            optimizer_name=optimizer_name,
        )
        # Tokenize datasets
        tokenizer = get_tokenizer(model_config.name)
        train_ds = tokenize_dataset(train_ds, tokenizer, max_ctx, weight = None)
        test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx, weight= None)
        if inference_ds:
            inference_ds = tokenize_dataset(inference_ds, tokenizer, max_ctx, weight=None)

        loss_fn = loss_dict[loss_type]

        return train_and_save_model(
            model_config,
            train_ds,
            test_ds,
            inference_ds=inference_ds,
            batch_size=batch_size,
            save_path=save_path,
            loss_fn=loss_fn,
            lr=lr,
            epochs=epochs,
            force_retrain=force_retrain,
            eval_batch_size=eval_batch_size,
            minibatch_size_per_device=minibatch_size_per_device,
            train_with_dropout=train_with_dropout,
            linear_probe=linear_probe,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
            eval_every=eval_every,
            is_weight=False,
            num_labels=1
        )

    # Train the weak model on the first half of the training data
    print(f"Training weak model, size {weak_model_size}")
    weak_test_results, weak_ds = train_model(
        weak_model_config,
        train1_ds,
        test_ds,
        loss_type="mse",
        label="weak",
        subpath=os.path.join("gradient_boost_weak_model", weak_model_size.replace("/", "_") + str(E)),
        lr=weak_lr,
        eval_batch_size=weak_eval_batch_size,
        inference_ds=train2_ds,
        epochs=gt_epochs,
        linear_probe=linear_probe,
        optimizer_name=weak_optim,
    )
    weak_acc = np.mean([x["acc"] for x in weak_test_results])
    res_dict = {
        "weak_acc": weak_acc,
    }
    print("weak acc:", weak_acc)
    with open(
        os.path.join(
            results_folder,
            f"gradient_10000_{weak_model_size.replace('/', '_')}{E}.results_summary.json",
        ),
        "w",
    ) as f:
        json.dump(
            res_dict,
            f,
        )

if __name__ == "__main__":
    fire.Fire(main)