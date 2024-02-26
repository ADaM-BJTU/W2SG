import itertools
import os
import pickle
import time
from dataclasses import dataclass
from typing import Callable, Optional

import datasets
import numpy as np
import torch
import torch_optimizer as toptim
from transformers.modeling_utils import load_sharded_checkpoint

import weak_to_strong.logger as logger
from weak_to_strong.common import clear_mem
from weak_to_strong.eval import eval_model_acc
from weak_to_strong.loss import xent_loss
from weak_to_strong.model import TransformerWithHead

from safetensors.torch import load_file


@dataclass
class ModelConfig:
    name: str
    default_lr: float
    eval_batch_size: int
    custom_kwargs: Optional[dict] = None
    gradient_checkpointing: bool = False
    model_parallel: bool = False
    default_optimizer: str = "adam"


def train_model(
        model: torch.nn.Module,
        ds: datasets.Dataset,
        batch_size: int,
        lr: float = 1e-5,
        loss_fn: Callable = xent_loss,
        log_every: int = 10,
        eval_every: int = 100,
        eval_batch_size: int = 256,
        minibatch_size: int = 8,
        eval_ds: Optional[datasets.Dataset] = None,
        gradient_checkpointing: bool = False,
        train_with_dropout: bool = False,
        epochs: int = 1,
        lr_schedule: str = "cosine_anneal",
        optimizer_name: str = "adam",
        save_path: str = None,
        test_ds: Optional[datasets.Dataset] = None,
        inference_ds: Optional[datasets.Dataset] = None,
        infer_valid_ds: Optional[datasets.Dataset] = None

):
    # if eval_ds:
    #     print(eval_ds)
    acc_list = [
        {
            "step": 0,
            "best_eval_acc": 0.0,
            "best_eval_test_acc": 0.0
        }
    ]
    print("LR", lr, "batch_size", batch_size, "minibatch_size", minibatch_size)
    assert batch_size % minibatch_size == 0, "batch size must be divisible by minibatch size"
    # we purposefully turn off dropout, for determinism
    # this seems to help for 1 epoch finetuning anyways
    if train_with_dropout:
        model.train()
    else:
        model.eval()
    if gradient_checkpointing:
        (
            model if hasattr(model, "gradient_checkpointing_enable") else model.module
        ).gradient_checkpointing_enable()

    nsteps = len(ds) * epochs // batch_size

    def lr_schedule_fn(step):
        if lr_schedule == "constant":
            return 1
        else:
            assert False, f"invalid lr schedule, {lr_schedule}, must be constant or cosine_anneal"

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "adafactor":
        optimizer = toptim.Adafactor(model.parameters(), lr=lr)
    else:
        assert False, f"invalid optimizer {optimizer_name}, must be adam or adafactor"
    if lr_schedule == "cosine_anneal":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nsteps)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    step = 0
    it = itertools.chain.from_iterable(itertools.repeat(ds, epochs))
    losses = []
    accuracies = []
    eval_acc_dict = {}

    # If the model is wrapped by DataParallel, it doesn't have a device. In this case,
    # we use GPU 0 as the output device. This sadly means that this device will store
    # a bit more data than other ones, but hopefully should not be too big of a deal.
    io_device = model.device if hasattr(model, "device") else 0
    best_eval_acc = 0.0
    while step < nsteps:
        loss_tot = 0
        if eval_ds and eval_every and step % eval_every == 0:
            print("Evaluating...")
            eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
            if gradient_checkpointing:
                (
                    model if hasattr(model, "gradient_checkpointing_enable") else model.module
                ).gradient_checkpointing_enable()
            if train_with_dropout:
                model.train()
            eval_accs = np.mean([r["acc"] for r in eval_results])
            acc_list.append(
                {
                    f"eval_acc_{step}": eval_accs
                }
            )
            eval_acc_dict[step] = eval_accs
            eval_save_path = os.path.join(save_path, str(step))
            if not os.path.exists(eval_save_path):
                os.makedirs(eval_save_path)
            if step != 0:
                (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
                    eval_save_path
                )
                torch.save(model.state_dict(), os.path.join(eval_save_path, "pytorch_model.bin"))
            with open(os.path.join(eval_save_path, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "avg_acc_eval": float(np.mean([r["acc"] for r in eval_results])),
                        "eval_results": eval_results,
                    },
                    f
                )
            if eval_accs > best_eval_acc and step != 0:
                best_eval_acc = eval_accs
                acc_list[0]["best_step"] = step
                acc_list[0]["best_eval_acc"] = best_eval_acc

                best_save_path = os.path.join(save_path, "eval_best_model")

                inference_results = None
                test_results = None
                infer_valid_results = None
                if not os.path.exists(best_save_path):
                    os.makedirs(best_save_path)
                if inference_ds:
                    print("infering...")
                    inference_results = eval_model_acc(model, inference_ds, eval_batch_size)
                if test_ds:
                    print("testing...")
                    test_results = eval_model_acc(model, test_ds, eval_batch_size)
                    best_eval_test_acc = float(np.mean([r["acc"] for r in test_results]))
                    acc_list[0]["best_eval_test_acc"] = best_eval_test_acc
                if infer_valid_ds:
                    print("infering on strong valid")
                    infer_valid_results = eval_model_acc(model, infer_valid_ds, eval_batch_size)
                    infer_valid_acc = float(np.mean([r["acc"] for r in infer_valid_results]))
                    acc_list[0]["best_infer_valid_acc"] = infer_valid_acc

                with open(os.path.join(best_save_path, "results.pkl"), "wb") as f:
                    pickle.dump(
                        {
                            "avg_acc_eval": float(np.mean([r["acc"] for r in eval_results])),
                            "avg_acc_inference": float(
                                np.mean([r["acc"] for r in inference_results] if inference_results else [])
                            ),
                            "avg_acc_test": best_eval_test_acc,
                            "avg_acc_infer_valid": infer_valid_acc if infer_valid_results else None,
                            "eval_results": eval_results,
                            "inference_results": inference_results if inference_results else [],
                            "test_results": test_results if test_results else [],
                            "infer_valid_results": infer_valid_results if infer_valid_results else [],
                        },
                        f,
                    )
                # tmp_save_path = "./tmp_result/gpt2/weak_model_gt/eval_best_model/model.safetensors"
                tmp_save_path =os.path.join(best_save_path,"model.safetensors")
                if os.path.isfile(tmp_save_path):
                    os.remove(tmp_save_path)
                (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
                    best_save_path
                )
                # tmp_save_path = "./tmp_result/gpt2/weak_model_gt/eval_best_model/pytorch_model.bin
                tmp_save_path = os.path.join(best_save_path, "pytorch_model.bin")
                if os.path.isfile(tmp_save_path):
                    os.remove(tmp_save_path)
                torch.save(model.state_dict(), os.path.join(best_save_path, "pytorch_model.bin"))
            print("Eval acc at step {}: {}".format(step, eval_accs))
            logger.logkvs(
                {
                    "step": step,
                    "eval_accuracy": eval_accs
                }
            )
        all_logits = []
        all_labels = []
        for i in range(batch_size // minibatch_size):
            try:
                mbatch = [next(it) for _ in range(minibatch_size)]
            except StopIteration:
                break
            input_ids = (
                torch.nn.utils.rnn.pad_sequence([torch.tensor(ex["input_ids"]) for ex in mbatch])
                .transpose(
                    0,
                    1,
                )
                .to(io_device)
            )
            labels = torch.tensor([ex["soft_label"] for ex in mbatch]).to(io_device)

            logits = model(input_ids)

            all_logits.extend(logits.to(io_device))
            all_labels.extend(labels)
        all_logits = torch.stack(all_logits)
        all_labels = torch.stack(all_labels)
        loss = loss_fn(all_logits, all_labels, step_frac=step / nsteps)
        loss_tot += loss.item()
        loss.backward()
        losses.append(loss_tot)
        accuracies.append(
            torch.mean(
                (torch.argmax(all_logits, dim=1) == torch.argmax(all_labels, dim=1)).to(
                    torch.float32
                )
            ).item()
        )
        logger.logkvs(
            {
                "step": step,
                "progress": step / nsteps,
                "loss": loss_tot,
                "train_accuracy": accuracies[-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        if log_every and step % log_every == 0:
            print(
                f"Step: {step}/{nsteps} Recent losses: {np.mean(losses)} {np.mean(accuracies)} {len(losses)}"
            )
            losses = []
            accuracies = []
        step += 1
        logger.dumpkvs()
    final_eval_results = None
    if eval_every:
        # 如果没有验证集就直接用测试集
        if eval_ds:
            print("Final evaluation:")
            final_eval_results = eval_model_acc(model, eval_ds, eval_batch_size)
        else:
            print("Final testing:")
            final_eval_results = eval_model_acc(model, test_ds, eval_batch_size)
        final_acc = np.mean([r["acc"] for r in final_eval_results])
        acc_list.append(
            {
                "eval_acc_last": final_acc
            }
        )
        logger.logkv("eval_accuracy", final_acc)
        logger.dumpkvs()
    return final_eval_results, acc_list


def train_and_save_model(
        model_config: ModelConfig,
        train_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        inference_ds: Optional[datasets.Dataset] = None,
        infer_valid_ds: Optional[datasets.Dataset] = None,
        *,
        batch_size: int,
        lr: float,
        epochs: int,
        eval_batch_size: Optional[int] = None,
        minibatch_size_per_device: Optional[int] = None,
        save_path: Optional[str] = None,
        loss_fn: Callable = xent_loss,
        label: str = "default",
        force_retrain: bool = False,
        train_with_dropout: bool = False,
        linear_probe: bool = False,
        lr_schedule: str = "constant",
        optimizer_name: str = "adam",
        eval_every: Optional[int] = None,
        eval_ds: Optional[datasets.Dataset] = None,
):
    if eval_batch_size is None:
        eval_batch_size = batch_size

    if minibatch_size_per_device is None:
        minibatch_size_per_device = 1

    gradient_checkpointing = model_config.gradient_checkpointing
    custom_kwargs = model_config.custom_kwargs or {}

    def maybe_load_model(model):
        if os.path.exists(os.path.join(save_path, "eval_best_model/results.pkl")) and not force_retrain:
            print("loading from", save_path)
            checkpoint_path = os.path.join(save_path, "eval_best_model/pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                # Assume this means we have a sharded checkpoint, and load it appropriately
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                # state_dict = torch.load(os.path.join(save_path, "eval_best_model/pytorch_model.bin"))
                # state_dict = {
                #     k.replace("transformer.module", "transformer"): v
                #     for (k, v) in state_dict.items()
                # }
                state_dict = torch.load(os.path.join(save_path, "eval_best_model/pytorch_model.bin"),
                                        map_location={'cuda:0': 'cuda:0', 'cuda:1': 'cuda:0'})
                custom_kwargs["state_dict"] = state_dict
                # state_dict=load_file(os.path.join(save_path, "eval_best_model/model.safetensors"))
                # custom_kwargs["state_dict"] = state_dict
            return True
        return False

    already_trained = False

    # Load the model
    if model_config.model_parallel:
        assert torch.cuda.device_count() > 1, f"you might want more gpus for {model_config.name}"
        model = TransformerWithHead.from_pretrained(
            model_config.name,
            num_labels=2,
            device_map="auto",
            linear_probe=linear_probe,
            **custom_kwargs,

        )
        already_trained = maybe_load_model(model)
        # slight misnomer, more like minibatch_size_per_dp_replica
        minibatch_size = minibatch_size_per_device
    else:
        model = TransformerWithHead.from_pretrained(
            model_config.name, num_labels=2, linear_probe=linear_probe, **custom_kwargs
        ).to("cuda")
        already_trained = maybe_load_model(model)
        # data parallel:  currently not supported with model parallel
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, output_device=0)
            minibatch_size = min(minibatch_size_per_device * torch.cuda.device_count(), batch_size)
            print(
                "Using",
                torch.cuda.device_count(),
                "GPUs, setting minibatch_size to",
                minibatch_size,
            )

    if already_trained:
        print("evaluating model on test set...")
        test_results = eval_model_acc(model, test_ds, eval_batch_size)
        final_save_path = os.path.join(save_path, "final_step")
        acc_list = [
            {
                "step": 0,
                "best_eval_acc": 0.0,
                "best_eval_test_acc": 0.0
            }
        ]
    else:
        start = time.time()
        eval_results, acc_list = train_model(
            model,
            train_ds,
            batch_size,
            eval_ds=eval_ds,
            lr=lr,
            epochs=epochs,
            test_ds=test_ds,
            gradient_checkpointing=gradient_checkpointing,
            loss_fn=loss_fn,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            # minibatch_size=minibatch_size,
            minibatch_size=batch_size,
            train_with_dropout=train_with_dropout,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
            save_path=save_path,
            inference_ds=inference_ds,
            infer_valid_ds=infer_valid_ds
        )
        print("Model training took", time.time() - start, "seconds")
        if save_path:
            final_save_path = os.path.join(save_path, "final_step")
            # Note: If the model is wrapped by DataParallel, we need to unwrap it before saving
            (model if hasattr(model, "save_pretrained") else model.module).save_pretrained(
                final_save_path
            )
            torch.save(model.state_dict(), os.path.join(final_save_path, "pytorch_model.bin"))
            print("saved", final_save_path)
    #
    inference_results = None
    infer_valid_results = None
    if inference_ds:
        print("Evaluating model on inference dataset...")
        inference_results = eval_model_acc(model, inference_ds, eval_batch_size)
        logger.logkv("inference_accuracy", np.mean([r["acc"] for r in inference_results]))
    if infer_valid_ds:
        print("Evaluating model on infer_valid dataset...")
        infer_valid_results = eval_model_acc(model, infer_valid_ds, eval_batch_size)
        # infer_valid_acc = float(np.mean([r["acc"] for r in infer_valid_results]))
        logger.logkv("infer_valid_accuracy", np.mean([r["acc"] for r in infer_valid_results]))
    if test_ds and not already_trained:
        print("Testing model on test dataset...")
        test_results = eval_model_acc(model, test_ds, eval_batch_size)
        test_acc = float(np.mean([r["acc"] for r in test_results]))
        if float(np.mean([r["acc"] for r in eval_results])) > acc_list[0]["best_eval_acc"]:
            acc_list[0]["best_eval_test_acc"] = test_acc
            acc_list[0]["best_step"] = -1
            acc_list[0]["best_eval_acc"] = -1
            if eval_ds:
                acc_list[0]["best_eval_acc"] = float(np.mean([r["acc"] for r in eval_results]))
                if infer_valid_ds:
                    acc_list[0]["best_infer_valid_acc"] = float(np.mean([r["acc"]for r in infer_valid_results]))
    if already_trained:
        acc_list[0]["best_test_eval_acc"] = float(np.mean([r["acc"] for r in test_results]))
    if not eval_ds and test_ds:
        acc_list[0] = {
            "test_acc": test_acc
        }

    if save_path and not already_trained:
        with open(os.path.join(final_save_path, "results.pkl"), "wb") as f:
            # 如果没有验证集，测试和验证一样
            pickle.dump(
                {
                    "avg_acc_eval": float(np.mean([r["acc"] for r in eval_results])),
                    "avg_acc_inference": float(
                        np.mean([r["acc"] for r in inference_results] if inference_results else [])
                    ),
                    "avg_acc_test": test_acc if test_results else None,
                    "avg_acc_infer_valid": float(np.mean([r["acc"] for r in infer_valid_results])) if infer_valid_results else [],
                    "eval_results": eval_results,
                    "inference_results": inference_results if inference_results else [],
                    "test_results": test_results,
                    "infer_valid_results": infer_valid_results,
                },
                f,
            )

    # try to clean up memory
    clear_mem()
    logger.shutdown()

    return test_results, inference_results, acc_list
