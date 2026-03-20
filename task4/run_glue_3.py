# coding=utf-8
# Task 4 (3): DDP with torch.profiler — profiles 3 training steps, skips first.
#
# Run on each node:
#   python run_glue_3.py [args] --master_ip <ip> --master_port <port> \
#       --world_size 4 --local_rank <0|1|2|3>
#
# After training, open trace_rank<N>_ddp.json in chrome://tracing

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
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """Train with DDP and torch.profiler (skip step 0, record steps 1-3)."""

    args.train_batch_size = args.per_device_train_batch_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.local_rank)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.module.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.module.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    trace_file = f"trace_rank{args.local_rank}_ddp.json"
    schedule = torch.profiler.schedule(wait=1, warmup=0, active=3, repeat=1)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    set_seed(args)
    profiling_done = False

    with torch.profiler.profile(
        schedule=schedule,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for epoch in range(int(args.num_train_epochs)):
            train_sampler.set_epoch(epoch)
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}",
                                              disable=args.local_rank not in [-1, 0])):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels':         batch[3]}
                outputs = model(**inputs)
                loss = outputs[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # DDP syncs grads automatically during backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                prof.step()

                if step >= 3:
                    profiling_done = True
                    break
            if profiling_done:
                break

    prof.export_chrome_trace(trace_file)
    if args.local_rank in [-1, 0]:
        print(f"Chrome trace saved to {trace_file}")

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
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels': batch[3]}
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
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0]:
        dist.barrier()
    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length), str(task)))
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']), cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0, sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']), pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
    if args.local_rank == 0:
        dist.barrier()
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--master_ip", default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="12345", type=str)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group(
        backend='gloo',
        init_method=f"tcp://{args.master_ip}:{args.master_port}",
        world_size=args.world_size, rank=args.local_rank)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory exists. Use --overwrite_output_dir.")

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    os.makedirs(args.output_dir, exist_ok=True)
    rank_log_file = os.path.join(args.output_dir, f"loss_rank{args.local_rank}.log")
    fh = logging.FileHandler(rank_log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    root_logger.setLevel(logging.INFO)
    if args.local_rank not in [-1, 0]:
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.WARN)

    set_seed(args)

    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    num_labels = len(processor.get_labels())

    if args.local_rank not in [-1, 0]:
        dist.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        dist.barrier()

    model.to(args.device)

    # Wrap with DDP
    model = DDP(model)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        train(args, train_dataset, model, tokenizer)

    if args.do_eval:
        evaluate(args, model.module, tokenizer, prefix="final")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
