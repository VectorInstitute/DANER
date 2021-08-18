import types
import sys
from copy import deepcopy, copy
from collections import defaultdict

import json
import scipy
import numpy as np

import torch
import transformers

from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments


import baal
from baal.active.active_loop import ActiveLearningLoop
from baal.active.dataset import ActiveLearningDataset, ActiveLearningPool
from baal.transformers_trainer_wrapper import BaalTransformersTrainer
from baal.active.heuristics.heuristics import AbstractHeuristic

use_cuda = torch.cuda.is_available()

print(use_cuda)
print(transformers.__version__)
print(torch.__version__)
print(baal.__version__)


def get_tokenized_dataset(tokenizer):
    datasets = load_dataset("conll2003")
    label_all_tokens = True

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
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

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)
    tokenized_datasets.set_format(
        columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_datasets


def compute_metrics(p, details=False):
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


class TokenALActiveLearningLoop(ActiveLearningLoop):
    """Object that perform the active learning iteration.
    Args:
        dataset (ActiveLearningDataset): Dataset with some sample already labelled.
        get_probabilities (Function): Dataset -> **kwargs ->
                                        ndarray [n_samples, n_outputs, n_iterations].
        heuristic (Heuristic): Heuristic from baal.active.heuristics.
        ndata_to_label (int): Number of sample to label per step.
        max_sample (int): Limit the number of sample used (-1 is no limit).
        **kwargs: Parameters forwarded to `get_probabilities`.
    """

    def step(self, pool=None) -> bool:
        """
        Perform an active learning step.
        Args:
            pool (iterable): dataset pool indices.
        Returns:
            boolean, Flag indicating if we continue training.
        """
        # High to low
        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(
                        len(pool), self.max_sample, replace=False)
                    pool = pool.select(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            probs = self.get_probabilities(pool, **self.kwargs)
            if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                to_label = self.heuristic(probs)
                if indices is not None:
                    to_label = indices[np.array(to_label)]
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.ndata_to_label])
                    return True
        return False


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


class MNLP(AbstractHeuristic):
    """
    Sort by the highest acquisition function value.
    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none`).
    References:
        https://arxiv.org/abs/1703.02910
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


class Margin(AbstractHeuristic):
    """
    Sort by the highest acquisition function value.
    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none`).
    References:
        https://arxiv.org/abs/1703.02910
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
            # Todo

        scores = np.array(scores)

        return self.reorder_indices(scores)


if __name__ == "__main__":
    output_dir = "random"

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    tokenized_datasets = get_tokenized_dataset(tokenizer)
    label_list = tokenized_datasets["train"].features["ner_tags"].feature.names

    model = AutoModelForTokenClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(label_list))
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")
    heuristic = MNLP(shuffle_prop=0.2)

    if use_cuda:
        model.cuda()
    init_weights = deepcopy(model.state_dict())

    active_set = HuggingFaceActiveLearningDataset(tokenized_datasets["train"])
    active_set.label_randomly(250)

    # Initialization for the huggingface trainer
    training_args = TrainingArguments(
        output_dir='./{}'.format(output_dir),  # output directory
        max_steps=10,  # total # of training steps per AL step
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./{}'.format(output_dir),  # directory for storing logs
    )

    # create the trainer through Baal Wrapper
    baal_trainer = TokenALTransformersTrainer(model=model,
                                              args=training_args,
                                              train_dataset=active_set,
                                              tokenizer=tokenizer,
                                              eval_dataset=tokenized_datasets["validation"],
                                              data_collator=data_collator,
                                              compute_metrics=compute_metrics)

    active_loop = TokenALActiveLearningLoop(active_set,
                                            baal_trainer.predict_on_dataset,
                                            heuristic, 250, iterations=1)

    res_dict = defaultdict(list)
    res = baal_trainer.evaluate(tokenized_datasets["test"])
    for k, v in res.items():
        res_dict[k].append(v)
    res_dict["train_num"].append(len(active_set))

    res_file = "al_mnlp_250_sh0.2.json"

    with open(res_file, 'w') as jsonfile:
        json.dump(res_dict, jsonfile)

    for epoch in range(3):
        baal_trainer.train()
        res = baal_trainer.evaluate(tokenized_datasets["test"])
        for k, v in res.items():
            res_dict[k].append(v)
        res_dict["train_num"].append(len(active_set))
        with open(res_file, 'w') as jsonfile:
            json.dump(res_dict, jsonfile)

        print(res_dict)
        sys.stdout.flush()

        # active_set.label_randomly(250)
        should_continue = active_loop.step()

        # We reset the model weights to relearn from the new train set.
        if not should_continue:
            break

        baal_trainer.load_state_dict(init_weights)
        baal_trainer.lr_scheduler = None
