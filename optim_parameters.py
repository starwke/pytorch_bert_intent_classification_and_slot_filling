import os
import time
import numpy as np
import optuna
import torch
from dataset import BertDataset
from model import BertForIntentClassificationAndSlotFilling
from config import Args
from optuna.trial import TrialState
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from preprocess import Processor, get_features
from torch.optim import Adam
from utils.logger import logger
from utils.metrics import get_metrices

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHES = 2
DIR = os.getcwd()

N_TRAIN_EXAMPLES = BATCH_SIZE * 30
N_TRAIN_EXAMPLES = BATCH_SIZE * 30


def define_model(trial, args: Args):
    args.device = DEVICE
    # 指定超参数
    # args.batchsize = trial.suggest_categorical("batchsize", [32, 64])
    args.lr = trial.suggest_float("lr", 2e-5, 1e-2, log=True)
    args.hidden_dropout_prob = trial.suggest_float("dropout", 0.2, 0.5)
    # args.max_len = trial.suggest_categorical("max_len", [20, 34, 40])
    model = BertForIntentClassificationAndSlotFilling(args)
    return model


def get_train_valid_data(args: Args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    raw_examples_train = Processor.get_examples(args.train_path, "train")
    train_features = get_features(raw_examples_train, tokenizer, args)
    train_dataset = BertDataset(train_features)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    # 测试数据
    raw_examples_test = Processor.get_examples(args.test_path, "test")
    test_features = get_features(raw_examples_test, tokenizer, args)
    test_dataset = BertDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)

    return train_loader, test_loader


def obejective(trial, args: Args):
    model = define_model(trial, args).to(DEVICE)
    train_loader, test_loader = get_train_valid_data(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(EPOCHES):
        start_time = time.time()
        for _, train_batch in enumerate(train_loader):
            for key in train_batch.keys():
                train_batch[key] = train_batch[key].to(args.device)
            input_ids = train_batch["input_ids"]
            attention_mask = train_batch["attention_mask"]
            token_type_ids = train_batch["token_type_ids"]
            seq_label_ids = train_batch["seq_label_ids"]
            token_label_ids = train_batch["token_label_ids"]
            seq_output, token_output = model(
                input_ids,
                attention_mask,
                token_type_ids,
            )

            active_loss = attention_mask.view(-1) == 1
            active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
            active_labels = token_label_ids.view(-1)[active_loss]

            seq_loss = criterion(seq_output, seq_label_ids)
            token_loss = criterion(active_logits, active_labels)
            loss = seq_loss + token_loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
        cost = time.time() - start_time
        logger.info("[train] epoch: {} time: {:.2f}s".format(epoch + 1, cost))

        model.eval()
        seq_preds = []
        seq_trues = []
        token_preds = []
        token_trues = []
        with torch.no_grad():
            for _, test_batch in enumerate(test_loader):
                for key in test_batch.keys():
                    test_batch[key] = test_batch[key].to(args.device)
                input_ids = test_batch["input_ids"]
                attention_mask = test_batch["attention_mask"]
                token_type_ids = test_batch["token_type_ids"]
                seq_label_ids = test_batch["seq_label_ids"]
                token_label_ids = test_batch["token_label_ids"]
                seq_output, token_output = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_label_ids = seq_label_ids.detach().cpu().numpy()
                seq_label_ids = seq_label_ids.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(seq_label_ids)

                token_output = token_output.detach().cpu().numpy()
                token_label_ids = token_label_ids.detach().cpu().numpy()
                token_output = np.argmax(token_output, -1)
                active_len = torch.sum(attention_mask, -1).view(-1)
                for length, t_output, t_label in zip(
                    active_len, token_output, token_label_ids
                ):
                    t_output = t_output[1 : length - 1]
                    t_label = t_label[1 : length - 1]
                    t_ouput = [args.id2nerlabel[i] for i in t_output]
                    t_label = [args.id2nerlabel[i] for i in t_label]
                    token_preds.append(t_ouput)
                    token_trues.append(t_label)

            acc, _, _, _ = get_metrices(seq_trues, seq_preds, "cls")

            ner_acc, _, _, _ = get_metrices(token_trues, token_preds, "ner")

            accuracy = np.mean([acc, ner_acc])
            trial.report(accuracy, epoch)
            # 判断是否满足剪枝条件，满足则抛出异常，中断当前实验
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        if args.do_save and accuracy >= 0.85:
            # 同时在日志中输出各个参数,并对模型进行保存
            logger.info(
                "[模型参数] lr: {} dropout: {:.2f}s acc:{:.2f} seqacc:{:.2f} ner_acc: {:.2f}".format(
                    args.lr, args.hidden_dropout_prob, accuracy, acc, ner_acc
                )
            )
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            save_name = f"model_{timestamp}.pt"
            torch.save(model.state_dict(), os.path.join(args.save_dir, save_name))
    return accuracy


if __name__ == "__main__":
    args = Args()
    storage_name = "sqlite:///optuna.db"
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        direction="maximize",
        study_name="man-machine-conversation",
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(lambda trial: obejective(trial, args), n_trials=20, timeout=3600)
