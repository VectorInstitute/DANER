import os
import sys
import json
import datetime
import types
from collections import defaultdict
from copy import deepcopy, copy

import torch
import scipy
import numpy as np

from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset, load_metric

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import DataCollatorForTokenClassification

from baal.active.dataset import ActiveLearningDataset, ActiveLearningPool
from baal.transformers_trainer_wrapper import BaalTransformersTrainer

from baal.active.heuristics.heuristics import AbstractHeuristic

use_cuda = torch.cuda.is_available()
print("CUDA availability: {}".format(use_cuda))
metric = load_metric("seqeval")

DEFAULT_LABEL = ['O', 'B-PER', 'I-PER', 'B-ORG',
                 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
BIO_MAP = {
    "B": 3,
    "I": 1,
    "O": 2
}
MAX_STEPS = 100
NUM_LABEL_RETRAIN_MODEL = 10
MAX_SAMPLES = 32
MAX_CANDIDATES = 32


def get_tokenized_dataset(tokenizer):
    datasets = load_dataset("conll2003")
    label_all_tokens = True

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        word_ids_list = []
        text = []
        annotators = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            word_ids_list.append(word_ids)
            text.append(" ".join(examples["tokens"][i]))
            annotators.append("default")

        tokenized_inputs["labels"] = labels
        tokenized_inputs["word_ids"] = word_ids_list
        tokenized_inputs["text"] = text
        tokenized_inputs["annotator"] = annotators
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets.set_format(
        columns=['input_ids', 'attention_mask', 'labels', 'word_ids', 'tokens'])
    return tokenized_datasets


def compute_metrics(p, label_list, details=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions, references=true_labels)

    if details:
        return results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


class HuggingFaceActiveLearningDataset(ActiveLearningDataset):
    @property
    def pool(self):
        """Returns a new Dataset made from unlabelled samples.
        Raises:
            ValueError if a pool specific attribute cannot be set.
        """
        pool_dataset = copy(self._dataset)

        for attr, new_val in self.pool_specifics.items():
            if hasattr(pool_dataset, attr):
                setattr(pool_dataset, attr, new_val)
            else:
                raise ValueError(f"{pool_dataset} doesn't have {attr}")

        pool_dataset = pool_dataset.select(
            (~self.labelled).nonzero()[0].reshape([-1]))
        ald = ActiveLearningPool(
            pool_dataset, make_unlabelled=self.make_unlabelled)
        return ald


class TokenALTransformersTrainer(BaalTransformersTrainer):
    def predict_on_dataset(self,
                           dataset,
                           iterations: int = 1,
                           half: bool = False,
                           ignore_keys=None):
        """
        Use the model to predict on a dataset `iterations` time.
        Args:
            dataset (Dataset): Dataset to predict on.
            iterations (int): Number of iterations per sample.
            half (bool): If True use half precision.
            ignore_keys (Optional[List[str]]): A list of keys in the output of your model
                (if it is a dictionary) that should be ignored when gathering predictions.
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.
        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        dataset = deepcopy(dataset._dataset)
        dataset.set_format(
            columns=['input_ids', 'attention_mask', 'labels'])
        preds = list(self.predict_on_dataset_generator(dataset=dataset,
                                                       iterations=iterations,
                                                       half=half,
                                                       ignore_keys=ignore_keys))

        seq_lens = [len(ex['attention_mask']) for ex in dataset]
        idx = 0
        probs = []

        for pred in preds:
            pred = scipy.special.softmax(pred.squeeze(-1), axis=-1)
            for i in range(pred.shape[0]):
                seq_len = seq_lens[idx+i]
                probs.append(pred[i][:seq_len])
            idx += pred.shape[0]
        return probs

    def get_train_dataloader(self):
        dataset = deepcopy(self.train_dataset)
        dataset._dataset.set_format(
            columns=['input_ids', 'attention_mask', 'labels'])
        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class MNLP(AbstractHeuristic):
    """
    Sort by the highest acquisition function value.
    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none`).
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=False, reduction=reduction
        )

    def get_ranks(self, predictions):
        """
        Compute the score according to the heuristic.
        Args:
            predictions (ndarray): Array of predictions
        Returns:
            Array of scores.
        """
        scores = []
        for pred in predictions:
            scores.append(np.mean(np.log(pred.max(-1))))

        scores = np.array(scores)

        return self.reorder_indices(scores)


class ALEngine():
    def __init__(self, model_name, label_list=DEFAULT_LABEL, need_train=False):
        # Hard Code the dataset and model
        self.label_list = label_list
        self.label_map = {label: idx for idx, label in enumerate(label_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name)
        self.hf_model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_list))

        self.init_weights = deepcopy(self.hf_model.state_dict())
        self.tokenized_datasets = get_tokenized_dataset(self.tokenizer)
        self.active_set = HuggingFaceActiveLearningDataset(
            self.tokenized_datasets["validation"])

        self.heuristic = MNLP(shuffle_prop=0.2)
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.training_args = TrainingArguments(
            # output directory
            output_dir='./tmp/{}'.format(
                datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
            max_steps=MAX_STEPS,  # total # of training steps per AL step
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./tmp/{}'.format(
                datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        )
        self.al_trainer = TokenALTransformersTrainer(model=self.hf_model,
                                                     args=self.training_args,
                                                     train_dataset=self.active_set,
                                                     tokenizer=self.tokenizer,
                                                     data_collator=self.data_collator,
                                                     compute_metrics=compute_metrics)
        self.get_probabilities = self.al_trainer.predict_on_dataset

        self.candidates = []   # maintain a list of example index for label
        self.new_label_count = 0
        self.max_sample = MAX_SAMPLES
        self.ndata_to_label = MAX_CANDIDATES

        # Log
        if not os.path.exists("log"):
            os.makedirs("log")
        self.log_file = "log/al{}".format(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self.res_dict = defaultdict(list)

        if need_train:
            self.active_set.label_randomly(1000)
            self.update_model()

        self.log_performance()
        self.update_candidates()
        print("AL Engine Initialized!")
        sys.stdout.flush()

    def log_performance(self):
        print("Log performance!")
        # res = self.al_trainer.evaluate(self.tokenized_datasets["test"])
        # for k, v in res.items():
        #     self.res_dict[k].append(v)
        # self.res_dict["train_num"].append(len(self.active_set))

        # with open(self.log_file, 'w') as jsonfile:
        #     json.dump(self.res_dict, jsonfile)

    def postprocess(self, token, word_ids, predictions):
        res = []
        previous_word_idx = None
        for word_idx, prediction in zip(word_ids, predictions[0]):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                pred = prediction.argmax(-1).numpy()
                label = self.label_list[pred]
                cur_label = "null" if pred == 0 else label[2:]
                iob = 2 if pred == 0 else BIO_MAP[label[0]]
                res.append({"token": token[word_idx], "label": cur_label, "iob": iob,
                           "confidence": prediction[pred].detach().numpy().tolist()})
            previous_word_idx = word_idx

        return res

    def next2label(self):
        if len(self.candidates) == 0:
            self.update_candidates()
        index = int(self.candidates[0])
        self.candidates = self.candidates[1:]

        inp = self.active_set._dataset[index]
        tokens = self.active_set._dataset["tokens"][index]
        word_ids = self.active_set._dataset["word_ids"][index]

        print({k: v for k, v in inp.items()})

        inp_cols = ['input_ids', 'attention_mask', 'labels']
        if use_cuda:
            inp = {k: torch.tensor(v).unsqueeze(0).to("cuda")
                   for k, v in inp.items() if k in inp_cols}
        else:
            inp = {k: torch.tensor(v).unsqueeze(0)
                   for k, v in inp.items() if k in inp_cols}

        pred = self.al_trainer.model(**inp).logits.softmax(-1).cpu()
        res = self.postprocess(tokens, word_ids, pred)
        return index, res

    def update_candidates(self):
        pool = self.active_set.pool
        if len(pool) > 0:
            # Limit number of samples
            if self.max_sample != -1 and self.max_sample < len(pool):
                indices = np.random.choice(
                    len(pool), self.max_sample, replace=False)
                pool._dataset = pool._dataset.select(indices)
            else:
                indices = np.arange(len(pool))

        probs = self.get_probabilities(pool)
        if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
            to_label = self.heuristic(probs)
            if indices is not None:
                to_label = indices[np.array(to_label)]
            if len(to_label) > 0:
                self.candidates = to_label[: self.ndata_to_label]
                return True

    def update_dataset(self, index, new_label, annotator="admin"):
        def dataset_map(example, idx):
            if idx != index:
                return {}
            else:
                label_all_tokens = True
                previous_word_idx = None
                label_ids = []
                for word_idx in example["word_ids"]:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(new_label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(
                            new_label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx

                if len(example["labels"]) != len(label_ids):
                    raise ValueError(
                        "Label Length Mismatch! Original: {} - New: {}".format(len(example["labels"]), len(label_ids)))
                return {"labels": label_ids, "annotator": annotator}

        new_label = self.process_new_label(new_label)
        self.active_set._dataset = self.active_set._dataset.map(
            dataset_map, with_indices=True)
        self.active_set.label([index])
        self.new_label_count += 1
        if self.new_label_count >= NUM_LABEL_RETRAIN_MODEL:
            self.update_model()
            self.new_label_count = 0
        return True

    def process_new_label(self, new_label):
        labels = []
        new_label = json.loads(new_label)

        for label in new_label:
            if label["label"] == "null":
                labels.append("O")
            else:
                labels.append(
                    "{}-{}".format("IOB"[label["iob"]-1], label["label"]))
        new_lab = [self.label_map[label] for label in labels]
        return new_lab

    def update_model(self):
        print("Retrain model from scratch!")
        sys.stdout.flush()
        self.al_trainer.load_state_dict(self.init_weights)
        self.al_trainer.train()
        self.log_performance()
