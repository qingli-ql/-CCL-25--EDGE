""" Finetuning the library models for sequence classification on GLUE."""

"""https://github.com/huggingface/transformers/blob/v4.9.0/examples/pytorch/text-classification/run_glue.py"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
from datasets import load_dataset
from typing import Optional, List
import transformers
from transformers import (
    GPT2ForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils.util import ensure_dir
from peft import LoraConfig, get_peft_model
import pandas as pd

from modules.EDGETrain import EDGETrainer
from modules.BaseTrainer import BaseTrainer
from modules.DataCollator import CustomDataCollator
from transformers import GPT2Config, GPT2Tokenizer

transformers.logging.set_verbosity_error()
os.environ["WANDB_DISABLED"] = "True"


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "snli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """

    task_name: Optional[str] = field(
        default="mnli",  # 任务名称默认设置为 "mnli"，你可以根据实际需要修改
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(["mnli", "snli", "qqp", "cola"])  # 可根据常见任务填写
        },
    )
    dataset_name: Optional[str] = field(
        default="glue",  # 默认使用GLUE数据集
        metadata={"help": "The name of the dataset (useful for naming things)."},
    )
    max_seq_length: int = field(
        default=128,  # 常见的最大序列长度，通常在NLP任务中使用
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,  # 默认不覆盖缓存
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    max_train_samples: Optional[int] = field(
        default=25600,  # 默认训练样本限制为5000
        metadata={"help": "Truncate the number of training examples to this value"},
    )
    max_eval_samples: Optional[int] = field(
        default=3200,  # 默认评估样本限制为1000
        metadata={"help": "Truncate the number of evaluation examples to this value"},
    )
    max_predict_samples: Optional[int] = field(
        default=1000,  # 默认预测样本限制为1000
        metadata={"help": "Truncate the number of prediction examples to this value"},
    )
    train_file: Optional[str] = field(
        default="data/biased_mnli/train.jsonl",  # 默认训练数据文件路径
        metadata={"help": "A json file for training data."}
    )
    validation_file: Optional[str] = field(
        default="data/biased_mnli/dev_match.jsonl",  # 默认验证数据文件路径
        metadata={"help": "A json file for validation data."}
    )
    test_file: Optional[str] = field(
        default="data/biased_mnli/dev_match.jsonl",  # 默认测试数据文件路径
        metadata={"help": "A json file for test data."}
    )
    mode: Optional[str] = field(
        default="CE", 
        metadata={"help": "train mode."}
    )
    alpha: Optional[float] = field(
        default=1.0, 
        metadata={"help": "train mode."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="./models/roberta-base",  # 默认使用roberta-large作为预训练模型
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default="./models/roberta-base",  # 默认使用roberta-large配置
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="./models/roberta-base",  # 默认使用roberta-large的tokenizer
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="outputs",  # 默认将预训练模型缓存到outputs目录
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )

@dataclass
class LoraArguments:
    """
    Arguments pertaining to LoRA.
    """

    lora_rank: int = field(default=16, metadata={"help": "Number of LoRA layers."})
    lora_alpha: int = field(
        default=32, metadata={"help": "Number of heads in LoRA."}
    )
    target_modules: Optional[List[str]] = field(default_factory=lambda: ["q", "k", "v"])
    lora_dropout: float = field(
        default=0.1, metadata={"help": "Dropout for LoRA."}
    )
    bias: Optional[str] = field(
        default="none", metadata={"help": "Bias for LoRA."}
    )
    task_type: Optional[str] = field(
        default="SEQ_CLS", metadata={"help": "Task type for LoRA."}
    )

def insertParams(model_args, data_args, training_args, lora_args):

    dataset_name = "SNLI_detail"
    dataset_name = "SNLI"

    model_args.model_name_or_path = "models/gpt2"
    train_file = f"data/biased_{dataset_name}/train.jsonl"
    eval_file = f"data/biased_{dataset_name}/dev.jsonl"
    test_file = f"data/biased_{dataset_name}/test.jsonl"

    data_args.train_file = train_file
    data_args.validation_file = eval_file
    data_args.test_file = data_args.test_file
    data_args.max_train_samples = 100000
    data_args.max_eval_samples = 2048 * 2

    
    training_args.label_names = ["labels", "type"]   
    training_args.remove_unused_columns = False   
    training_args.include_inputs_for_metrics = True   

    # 设置训练相关的其他参数
    # training_args.output_dir = "outputs/biased_SNLI"
    # training_args.device = torch.device("cuda:1") 
    training_args.do_train = True
    training_args.do_eval = True
    training_args.num_train_epochs = 15
    training_args.learning_rate = 3e-4 
    training_args.warmup_ratio = 0.06
    training_args.weight_decay = 0.1
    training_args.warmup_steps = 500
    training_args.per_device_train_batch_size = 64
    training_args.per_device_eval_batch_size = 128
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "epoch"
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 500
    training_args.dataloader_num_workers = 4
    training_args.seed = 2023
    training_args.overwrite_cache = True
    # best model
    training_args.load_best_model_at_end = True 
    training_args.metric_for_best_model = "accuracy" 
    training_args.greater_is_better = True  
    
    return model_args, data_args, training_args, lora_args

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, lora_args = insertParams(model_args, data_args, training_args, lora_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None

    # Set seed before initializing model
    set_seed(training_args.seed)

    data_files = {}
    if training_args.do_train:
        data_files["train"] = data_args.train_file
    if training_args.do_eval:
        data_files["validation"] = data_args.validation_file
    if training_args.do_predict:
        data_files["test"] = data_args.test_file

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    raw_datasets = load_dataset(
        "json", data_files=data_files, cache_dir=model_args.cache_dir
    )
    sentence1_key, sentence2_key, label_key = "sentence1", "sentence2", "label"

    # labels
    label_list = ["contradiction", "entailment", "neutral"]
    # label_list = ["entailment", "not_entailment"]
    label_list.sort()

    def filter_function(example):
        valid_labels = set(label_list)
        return example[label_key] in valid_labels

    raw_datasets = raw_datasets.filter(filter_function)


    # config = RobertaConfig.from_pretrained(
    #     (
    #         model_args.model_name_or_path
    #     ),
    #     cache_dir=model_args.cache_dir,
    #     num_labels=len(label_list),
    # )
    # tokenizer = RobertaTokenizer.from_pretrained(
    #     (
    #         model_args.model_name_or_path
    #     ),
    #     cache_dir=model_args.cache_dir,
    #     use_fast=True,
    # )
    # model = RobertaForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    config = GPT2Config.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3,
        cache_dir=None  # 如果不需要缓存，可以忽略
    )

    # 加载分词器
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=None  # 如果不需要缓存，可以忽略
    )
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # 加载模型
    model = GPT2ForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=None  # 如果不需要缓存，可以忽略
    )
    model.config.pad_token_id = model.config.eos_token_id
    
    
    lora_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1) 
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 
    type_list = pd.Series(raw_datasets['validation']['type']).unique()

    type2id = {v: i for i, v in enumerate(type_list)}
    id2type = {id: type for type, id in type2id.items()}
    label_to_id = {v: i for i, v in enumerate(label_list)}
    id2label = {id: label for label, id in label_to_id.items()}
    model.config.label2id = label_to_id
    model.config.id2label = id2label
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    first = True
    def preprocess_function(examples):
        args = (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(
            *args, padding="max_length", max_length=max_seq_length, truncation=True, return_tensors="pt"
        )
        if not training_args.do_predict:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples[label_key]
            ]
        result['type'] = [type2id[_type] for _type in examples['type']]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            num_proc=12,
        )
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples)
            )

    # log some random samples from the training set
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(predictions, axis=1)
        labels, types = p.label_ids
    
        correct = (preds == labels).astype(np.float32)
        overall_accuracy = correct.mean().item()
        eval_df = pd.DataFrame({"label": labels, "correct": correct, "type": types})
        eval_df['type'] = eval_df['type'].map(id2type)
        type_accuracy = eval_df.groupby("type")["correct"].mean().to_dict()
        class_accuracy = eval_df.groupby("label")["correct"].mean().to_dict()
        class_accuracy = {id2label[k]: v for k, v in class_accuracy.items()}
        print("--------------------------------------")
        print("-----------eval-metric-------------->>")
        print(f"  Overall accuracy: {overall_accuracy:.3f}")
        print(f"  Each type accuracy: ", {f"{k}_accuracy": f"{v:.3f}" for k, v in type_accuracy.items()})
        print(f"  Each class accuracy: ", {f"{k}_accuracy": f"{v:.3f}" for k, v in class_accuracy.items()})
        return {
            "accuracy": overall_accuracy,
            # **{f"{k}_accuracy": v for k, v in type_accuracy.items()},
            # **{f"{k}_accuracy": v for k, v in class_accuracy.items()},
        }
    data_collator = CustomDataCollator(tokenizer=tokenizer, padding=True)
    Mytrainer = BaseTrainer
    if data_args.mode == 'GOD':
        Mytrainer = EDGETrainer
        
    trainer = Mytrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.setMode(data_args.mode, data_args.alpha)
    # train!
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_df = pd.DataFrame(eval_dataset)
        predictions = trainer.predict(
            eval_dataset, metric_key_prefix="predict"
        ).predictions
        predictions = np.argmax(predictions, axis=1)
        eval_df['pred'] = predictions
        eval_df['correct'] = eval_df['label'] == eval_df['pred']
        eval_df['type'] = eval_df['type'].map(id2type)
        overall_acc = eval_df['correct'].mean()
        print(f"Overall accuracy: {overall_acc}")
        each_acc = eval_df.groupby('type')['correct'].mean().to_dict()
        print(f"Each type accuracy: {each_acc}")

        each_acc = {id2label[k]:v for k, v in eval_df.groupby('label')['correct'].mean().to_dict().items()}
        print(f"Each class accuracy: {each_acc}")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics(data_args.dataset_name, metrics)
        trainer.save_metrics(data_args.dataset_name, metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = predict_dataset.remove_columns(label_key)
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        ).predictions
        predictions = np.argmax(predictions, axis=1)

        output_dir = os.path.join(training_args.output_dir, "predictions")
        ensure_dir(output_dir)
        output_predict_file = os.path.join(
            output_dir, f"{data_args.dataset_name}_predictions.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
