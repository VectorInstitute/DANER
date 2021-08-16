from typing import List
from copy import deepcopy

import spacy
import torch

use_cuda = torch.cuda.is_available()
DEFAULT_LABEL = ['O', 'B-PER', 'I-PER', 'B-ORG',
                 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
BIO_MAP = {
    "B": 3,
    "I": 1,
    "O": 2
}

nlp = spacy.load("en_core_web_sm")


class HFModel():
    def __init__(self, tokenizer, model, label_list=DEFAULT_LABEL):
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.model = model
        self.init_weights = deepcopy(self.model.state_dict())

        if use_cuda:
            self.model.cuda()

    def tokenize(self, s):
        # Make the tokenization consistent with spacy
        if not isinstance(s, List):
            ls = [t.text for t in nlp(s) if t.text.strip()]
            merge = [(i-1, i+2)
                     for i, s in enumerate(ls) if i >= 1 and s == '-']
            for t in merge[::-1]:
                merged = ''.join(ls[t[0]:t[1]])
                ls[t[0]:t[1]] = [merged]
            text = ls
        else:
            text = s

        # text is a list of raw tokens.
        # Get Input to the transformer model and maitain the information about subtoken
        inp = self.tokenizer(text, return_tensors="pt",
                             is_split_into_words=True)
        word_ids = inp.word_ids(batch_index=0)

        if use_cuda:
            inp = {k: v.to('cuda') for k, v in inp.items()}

        return text, inp, word_ids

    def postprocess(self, text, word_ids, predictions, confidence=0.2):
        res = []
        previous_word_idx = None
        for word_idx, prediction in zip(word_ids, predictions[0]):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                pred = prediction.argmax(-1).numpy()
                if pred != 0:
                    if prediction[pred] < confidence:
                        pred = 0
                cur_label = self.label_list[pred]
                res.append({"token": text[word_idx], "label": "null" if pred ==
                           0 else cur_label[2:], "iob": BIO_MAP[cur_label[0]]})
            previous_word_idx = word_idx

        return res

    def predict(self, text, confidence=0.2):
        # Does not support batch prediction, text can be string or token list
        text, inp, word_ids = self.tokenize(text)
        pred = self.model(**inp).logits.softmax(-1).cpu()
        res = self.postprocess(text, word_ids, pred, confidence=confidence)
        return res
