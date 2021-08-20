---
theme: seriph
title: Interim Presentation

highlighter: shiki

colorSchema: "light"

info: |
  ## Slidev
  Vector Summer Internship Project Interim Presentation

background: vector_cover.png
layout: cover
class: text-center
---

# Build a Customized NLP Service

## Interim Presentation

Friday, July 9, 2021

Presented by: Yongchao Zhou

---

# Motivation

<div class="wrapper gap-4">
<div class="pt-2"> 
<img src="ml_workflow.png" class="max-w-xl m-auto">
<figcaption class="text-center pt-4"> Figure 1: Machine Learning Life Cycle</figcaption>
</div>

<div class="pt-2">

- <div class="text-xl py-4">ML Application Life Cycle</div>

  - <div>Data Collection</div>
  - <div>Data Annotation</div>
  - <div>Model Training & Evaluation</div>
  - <div>Model Deployment</div>

- <div class="text-xl py-4">ML Tools</div>

  - Active Learning
  - Transfer Learning

</div>
</div>

<v-click>
<div class="border-red-600 border-4 absolute h-29 w-34 left-51 top-31 "></div>
<div class="border-red-600 border-4 absolute h-17 w-34 left-87 top-89 "></div>
</v-click>

<style>
.wrapper {
  display: grid;
  grid-template-columns: 1.5fr 2fr;
}
</style>

<!--
Notes: Here, I show a typical life cycle of a ML application, where we have four different phases, namely, data collection, data annotation, model training & evaluation and finally model deployment. For many real world ML application, data annotation can be a big headache, especially for the domain where a high level expertise is needed, such as medical, law and finance. So, the motivation question of this project is how can we build an annotation-efficient Machine Learning System?

In this project, we will utilize the recent advancements in two areas of machine learning, one is active learning, the other is transfer learning. Those two techniques address the problem when we have limited number of labeled data from two different perspectives. The goal of active learning is to identify the most informative examples from the unlabeled dataset for human to label. In contrast, the transfer learning aims to apply the knowledge from one domain to another. 

To demonstrate the power of combining those two techniques in a real world ML system, we create this project to build a customized NER service from scratch.
-->

---

# NER service

<div grid="~ cols-2 gap-x-10 gap-y-2">
<div>

- Inference API
</div>

- Graphical User Interface

<div v-click>

```python{all}
>>> import NerModel
>>> model = NerModel(args.model_dir)
>>> model(args.text)

{'text': 'Geoffrey Everest Hinton (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks.',
'ents': [{'token': 'Geoffrey Everest Hinton', 'start': 0, 'end': 23, 'label': 'PERSON'},
        {'token': '6', 'start': 30, 'end': 31, 'label': 'CARDINAL'},
        {'token': 'December 1947', 'start': 32, 'end': 45, 'label': 'DATE'},
        {'token': 'British', 'start': 52, 'end': 59, 'label': 'NORP'}]}
```

</div>

<div v-click>

<img src="ner_service.png" width=500>
</div>
</div>

<style>
.wrapper2 {
  display: grid;
  grid-template-columns: 1.5fr 2fr;
}
</style>

<!--
Notes: So, what I mean by customized NER service. Basically, it is a machine learning system trained on some domain-specific datasets. It provides the clients with two things. One is an inference API that the user can call to analyze a sentence. The other is a graphical user interface. We show the API usage on the left figure, which you can call in your command line or using python scripts. In contrast, the right figure shows a graphical user interface. When users come to use it, they can just type the sentence and the system will give the well-rendered result back. Moreover, they can also choose which labels they are interested in as long as it is within the model's capability and the system will render the outputs differently.
-->

---

# How to build a customized NER service?

- Data Collection -> Data Annotation -> Model Training & Evaluation -> Model Deployment (API/GUI)

<div v-click="1">
<div class="border-red-600 border-4 absolute h-10 w-35 left-55 top-23 "></div>
</div>

<div v-click="2">
```python {all}
>>> raw_dataset[0]
"Geoffrey Everest Hinton (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks."
```
</div>

<arrow v-click="2" x1="520" y1="185" x2="520" y2="243" color="#dc2626" width="2" arrowSize="1"> </arrow>

<div v-click="2" class="absolute left-14 top-61 w-217">
```python {all}
>>> train_dataset[0]
{'text': 'Geoffrey Everest Hinton (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks.',
'annotation': [{'token': 'Geoffrey', 'label': 'PERSON', 'iob': 3}, {'token': 'Everest', 'label': 'PERSON', 'iob': 1}, {'token': 'Hinton', 'label': 'PERSON', 'iob': 1}, {'token': '(', 'label': 'null', 'iob': 2},
         {'token': 'born', 'label': 'null', 'iob': 2}, {'token': '6', 'label': 'CARDINAL', 'iob': 3}, {'token': 'December', 'label': 'DATE', 'iob': 3}, {'token': '1947', 'label': 'DATE', 'iob': 1},
         {'token': ')', 'label': 'null', 'iob': 2}, {'token': 'is', 'label': 'null', 'iob': 2}, {'token': 'a', 'label': 'null', 'iob': 2}, {'token': 'British', 'label': 'NORP', 'iob': 3},
         {'token': '-', 'label': 'null', 'iob': 2}, {'token': 'Canadian', 'label': 'null', 'iob': 2}, {'token': 'cognitive', 'label': 'null', 'iob': 2}, {'token': 'psychologist', 'label': 'null', 'iob': 2},
         {'token': 'and', 'label': 'null', 'iob': 2}, {'token': 'computer', 'label': 'null', 'iob': 2}, {'token': 'scientist', 'label': 'null', 'iob': 2}, {'token': ',', 'label': 'null', 'iob': 2},
         {'token': 'most', 'label': 'null', 'iob': 2}, {'token': 'noted', 'label': 'null', 'iob': 2}, {'token': 'for', 'label': 'null', 'iob': 2}, {'token': 'his', 'label': 'null', 'iob': 2},
         {'token': 'work', 'label': 'null', 'iob': 2}, {'token': 'on', 'label': 'null', 'iob': 2}, {'token': 'artificial', 'label': 'null', 'iob': 2}, {'token': 'neural', 'label': 'null', 'iob': 2},
         {'token': 'networks', 'label': 'null', 'iob': 2}, {'token': '.', 'label': 'null', 'iob': 2}
         ... ]}
```
</div>

<!--
Notes: So, how to build a customized NER service from scratch? Basically, we need to complete all stages of ML applcaition ourself. Start from data collection and labeling to model training, evaluation and deployment. In the previous slides, we have seen what does the model deployment look like. Now, let's have a closer look into the data annotation part. The goal of data annotation is to transform the raw_dataset to the annotated dataset as shown in the transformation above, where the raw dataset consists a list of string. In contrast, each example in the annotated dataset is a dictionary with the label information. To achieve this transformation, We introduce our data annotation interface.
-->

---

# Annotation Interface

<div class="wrapper gap-1">
<div class="place-self-center">
<img v-click src="annotation_setup.png" class="max-h-96">
</div>

<div class="place-self-center">
<img v-click src="annotation_interface.png" class="max-h-96 ">
</div>

</div>

<style>
.wrapper {
  display: grid;
  grid-template-columns: 1.5fr 2fr;
  grid-auto-rows: minmax(min-content, max-content)
}
</style>

<!--
Notes: The data annotation interface consists of two parts, one is the configuration, the other is the labeling. Before annotating, we need to do some setups, such as upload the dataset, define the label set and configure the annotation parameters. Specifically, we may want to tune the auto suggestion strength and the active learning algorithm. After that, we can start annotate the data using our interface. It's pretty easy to work with, you can just select a label and then click on the token you want to assign the label for. To remove a label, you can simply click the token. After you finish, you click the checkmark. If you don't want to label some examples at the moment. You can either skip it and go back later or just reject it so that it will not be in the training set.
-->

---

# Data Quality Control (In Progress)

<div grid="~ cols-2 gap-10">

<div>
```python {all|4-6,8,11-13,15}
>>> db.examples[0]
{'text': 'Geoffrey Everest Hinton (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks.', 
'annotations': [{
  'user1':[{'token': 'Geoffrey', 'label': 'PERSON', 'iob': 3}, 
        {'token': 'Everest', 'label': 'PERSON', 'iob': 1}, 
        {'token': 'Hinton', 'label': 'PERSON', 'iob': 1}, 
        {'token': 'born', 'label': 'null', 'iob': 2}, 
        {'token': '6', 'label': 'CARDINAL', 'iob': 3}, 
        {'token': 'December', 'label': 'DATE', 'iob': 3}, 
        ... ],
  'user2':[{'token': 'Geoffrey', 'label': 'null', 'iob': 2}, 
        {'token': 'Everest', 'label': 'null', 'iob': 2}, 
        {'token': 'Hinton', 'label': 'null', 'iob': 2}, 
        {'token': 'born', 'label': 'null', 'iob': 2}, 
        {'token': '6', 'label': 'DATE', 'iob': 3}, 
        {'token': 'December', 'label': 'DATE', 'iob': 3}, 
        ... ]}]
```
</div>

<div>
<div v-click="2">

- User 1
  <img src="disagree_1.png">

- User 2
  <img src="disagree_2.png">

</div>

<div v-click="3">

- Disagreement Visualization
  - Help needed

</div>

<div v-click="4">

- Golden Standard
  <img src="disagree_3.png">

</div>

</div>

</div>

<!--
Notes: Just labeling a dataset is not enough for a data annotation tool. Another key feature of it is to resolve the annotation disagreement. For example, if we have two annotations from two different users and they do not agree with each other. The user 1 believes that Geoffrey Hinton is a person, while the second user does not. And the first user think the 6 is a simple cardinal number, but the user 2 think it is actually part of a date. Therefore, we must need to resolve these conflicts. Basically, we need to somehow visualize the disagreement and then correct them to form a golden standard annotation. But currently, I still not sure how I should visualize the disagreement, especially if there are more the two annotators.
-->

---

# Model Training

```python {all|1|2-8|10|12-17|19|all}
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
)

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
```

<!--
Notes: After we get our golden standard labeled dataset, we are able to train our model! We will rely on the hugging face transformer library. We can initialize from a pretrained model and perform the standard finetune process. Here is where the transfer learning kicks in.
-->

---

# Model Selection using A/B Test (In progress)

<div v-click>

- Prediction from Trained Model 1
  <img src="disagree_1.png">

- Prediction from Trained Model 2
  <img src="disagree_2.png">
</div>

<div v-click>

- They both make one mistake and have the same accuracy. Which model to choose?
  - Let user make the decision. 
  - E.g. A/B Test with 100 examples. 
    - Model 1 has 68/100 "likes" and Model 2 has 55/100 "likes" -> Deploy model 1

</div>

<!--
Notes: After you have trained several models and you may find all those models achieve the same level of accuracy, then you may face the question of which model should I choose for deployment. For example, now we have trained two models, and the predictions from each model are shown above. Note that they are not annotations from two different users, instead they are just two predictions. Now, we may perform a A/B test and let the user to do the selection. For example, we can give the user a example with two prediction, and the user can choose which prediction they prefer. Then, we repeat the process for 100 examples, we count how many times the user prefer the model 1's prediction and how many times the user prefer the model 2's prediction and we will select the model with more likes for deployment.
-->

---

# Next Steps

- Functionality
  - Complete the data quality control interface
  - Model Selection with A/B testing
  - Data annotation with active learning
- Experiement
  - Benchmark the performance of different active learning algorithm on NER task
- Demo
  - Build a customized NER service for a specific domain (Medical Data)

<!--
Notes: So, that is all my showing for today, there are some features still in progress, such as data quality control interface, model selection with A/B test interface and most importantly, the data annotation with active learning. After complete those, I also need to benchmark the performance of different active learning algorithm on NER task. So, I know which active learning algorithm to use. And finally, I will build a customized NER service for a specifc domain, may be medical data? And I will need some help from the team for this part. That's the end of my presentation. Thank you everyone for listening.
-->
