# coding=utf-8
# Unified run_glue.py for COS 568 Assignment 2.
#
# Controls all tasks via --sync_method and --profile:
#   Task 1:  --sync_method none           (single-node)
#   Task 2a: --sync_method gather_scatter  (distributed, gather/scatter)
#   Task 2b: --sync_method all_reduce      (distributed, all_reduce)
#   Task 3:  --sync_method ddp             (distributed, DistributedDataParallel)
#   Task 4:  add --profile to any of Task 2a/2b/3

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert':    (BertConfig,    BertForSequenceClassification,    BertTokenizer),
    'xlnet':   (XLNetConfig,   XLNetForSequenceClassification,   XLNetTokenizer),
    'xlm':     (XLMConfig,     XLMForSequenceClassification,     XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def is_distributed(args):
    return args.sync_method != 'none'


def sync_gradients(model, args):
    """Synchronize gradients according to the chosen sync method."""
    if args.sync_method == 'gather_scatter':
        for param in model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.data
            if args.local_rank == 0:
                gather_list = [torch.zeros_like(grad) for _ in range(args.world_size)]
                dist.gather(grad, gather_list=gather_list, dst=0)
                avg_grad = torch.stack(gather_list).mean(dim=0)
                scatter_list = [avg_grad.clone() for _ in range(args.world_size)]
                dist.scatter(grad, scatter_list=scatter_list, src=0)
            else:
                dist.gather(grad, dst=0)
                dist.scatter(grad, src=0)
            param.grad.data = grad

    elif args.sync_method == 'all_reduce':
        for param in model.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= args.world_size

    # 'ddp' and 'none': no manual sync needed


def run_training_step(step, batch, model, optimizer, scheduler, args):
    """Execute one training step; return loss value."""
    model.train()
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {
        'input_ids':      batch[0],
        'attention_mask': batch[1],
        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
        'labels':         batch[3],
    }
    outputs = model(**inputs)
    loss = outputs[0]

    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    # Manual gradient sync for gather_scatter and all_reduce
    # (DDP syncs automatically during backward; 'none' is single-node)
    raw_model = model.module if args.sync_method == 'ddp' else model
    sync_gradients(raw_model, args)

    if (step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    return loss.item()


def train(args, train_dataset, model, tokenizer):
    """Main training loop."""
    args.train_batch_size = args.per_device_train_batch_size

    if is_distributed(args):
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=args.world_size,
                                           rank=args.local_rank)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (args.max_steps //
                                 (len(train_dataloader) // args.gradient_accumulation_steps) + 1)
    else:
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps
                   * args.num_train_epochs)

    no_decay = ['bias', 'LayerNorm.weight']
    param_source = model.module if args.sync_method == 'ddp' else model
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_source.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_source.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    is_main = args.local_rank in [-1, 0]
    if is_main:
        total_batch = args.per_device_train_batch_size * (args.world_size if is_distributed(args) else 1)
        logger.info("***** Running training [%s] *****", args.sync_method)
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Per-device batch size = %d", args.per_device_train_batch_size)
        logger.info("  Total batch size = %d", total_batch)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    iter_times = []
    model.zero_grad()
    set_seed(args)

    def _train_loop(prof=None):
        nonlocal global_step, tr_loss
        for epoch in range(int(args.num_train_epochs)):
            if is_distributed(args):
                train_sampler.set_epoch(epoch)
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                                  disable=not is_main)
            for step, batch in enumerate(epoch_iterator):
                iter_start = time.time()

                loss_val = run_training_step(step, batch, model, optimizer, scheduler, args)
                tr_loss += loss_val
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1

                iter_end = time.time()
                if step > 0:  # skip first iteration timing
                    iter_times.append(iter_end - iter_start)

                # Task 1: print loss for first 5 minibatches
                if args.sync_method == 'none' and step < 5:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss_val:.6f}")

                if is_main:
                    logger.info("Epoch %d, Step %d, Loss: %.6f", epoch, step, loss_val)

                if prof is not None:
                    prof.step()

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                break

            # Evaluate after each epoch
            eval_model = model.module if args.sync_method == 'ddp' else model
            if is_main:
                evaluate(args, eval_model, tokenizer)

    if args.profile:
        trace_file = f"trace_rank{max(args.local_rank, 0)}_{args.sync_method}.json"
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=3, repeat=1)
        with torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('.'),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            _train_loop(prof=prof)
        prof.export_chrome_trace(trace_file)
        if is_main:
            print(f"Chrome trace saved to {trace_file}")
    else:
        _train_loop()

    if iter_times and is_distributed(args) and is_main:
        avg = sum(iter_times) / len(iter_times)
        logger.info("Average iteration time (excluding first): %.4f s", avg)
        print(f"[Rank {args.local_rank}] Avg iteration time (excl. first): {avg:.4f} s")

    return global_step, tr_loss / max(global_step, 1)


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation %s *****", prefix)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels':         batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results %s *****", prefix)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if is_distributed(args) and args.local_rank not in [-1, 0]:
        dist.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (processor.get_dev_examples(args.data_dir) if evaluate
                    else processor.get_train_examples(args.data_dir))
        features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if is_distributed(args) and args.local_rank == 0:
        dist.barrier()

    all_input_ids  = torch.tensor([f.input_ids  for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--data_dir",           required=True, type=str)
    parser.add_argument("--model_type",          required=True, type=str)
    parser.add_argument("--model_name_or_path",  required=True, type=str)
    parser.add_argument("--task_name",           required=True, type=str)
    parser.add_argument("--output_dir",          required=True, type=str)

    # Task selection
    parser.add_argument("--sync_method", default="none",
                        choices=["none", "gather_scatter", "all_reduce", "ddp"],
                        help="Gradient sync method. 'none'=single node (Task 1), "
                             "'gather_scatter'=Task 2a, 'all_reduce'=Task 2b, 'ddp'=Task 3.")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler (Task 4). Skips step 0, records 3 steps.")

    # Distributed
    parser.add_argument("--master_ip",   default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="12345",      type=str)
    parser.add_argument("--world_size",  default=1,            type=int)
    parser.add_argument("--local_rank",  default=-1,           type=int)

    # Model / data
    parser.add_argument("--config_name",    default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir",      default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train",       action="store_true")
    parser.add_argument("--do_eval",        action="store_true")
    parser.add_argument("--do_lower_case",  action="store_true")

    # Training
    parser.add_argument("--per_device_train_batch_size", default=8,    type=int)
    parser.add_argument("--per_device_eval_batch_size",  default=8,    type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1,    type=int)
    parser.add_argument("--learning_rate",               default=5e-5, type=float)
    parser.add_argument("--weight_decay",                default=0.0,  type=float)
    parser.add_argument("--adam_epsilon",                default=1e-8, type=float)
    parser.add_argument("--max_grad_norm",               default=1.0,  type=float)
    parser.add_argument("--num_train_epochs",            default=3.0,  type=float)
    parser.add_argument("--max_steps",                   default=-1,   type=int)
    parser.add_argument("--warmup_steps",                default=0,    type=int)
    parser.add_argument("--no_cuda",            action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache",    action="store_true")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--fp16",               action="store_true")
    parser.add_argument("--fp16_opt_level",     default="O1", type=str)

    args = parser.parse_args()

    # Initialize distributed process group for Tasks 2a/2b/3/4
    if is_distributed(args):
        dist.init_process_group(
            backend='gloo',
            init_method=f"tcp://{args.master_ip}:{args.master_port}",
            world_size=args.world_size,
            rank=args.local_rank,
        )

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists. Use --overwrite_output_dir.")

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Rank: %s | Device: %s | sync_method: %s | profile: %s",
                   args.local_rank, args.device, args.sync_method, args.profile)

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Only rank 0 downloads model weights; others wait at barrier
    if is_distributed(args) and args.local_rank not in [-1, 0]:
        dist.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if is_distributed(args) and args.local_rank == 0:
        dist.barrier()

    model.to(args.device)

    # Wrap with DDP for Task 3 (and Task 4 DDP variant)
    if args.sync_method == 'ddp':
        model = DDP(model)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval and args.local_rank in [-1, 0]:
        eval_model = model.module if args.sync_method == 'ddp' else model
        evaluate(args, eval_model, tokenizer, prefix="final")

    if is_distributed(args):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
