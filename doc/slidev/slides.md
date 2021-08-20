---
theme: seriph
title: Interim Presentation

highlighter: shiki

colorSchema: "light"

info: |
  ## Slidev
  Vector Summer Internship Project Final Presentation

background: vector_cover.png
layout: cover
class: text-center
download: true
---

# Build a Data Annotation Tool using AL and TL

## Final Presentation

Thursday, August 19, 2021

Presented by: Yongchao Zhou

<!--
Hi everyone, welcome to my presentation. My name is Yongchao, I am an applied machine learning engineer intern at Vector. My summer project can be summarized as building a data annotation tool using active learning and transfer learning.
-->

---

# Motivation

<br>
<br>

<img src="final_ml_lifecycle.jpg" class="max-w-xl m-auto" width="850">
<figcaption class="text-center pt-5 text-lg"> Figure 1: Machine Learning Life Cycle</figcaption>

<v-click>
<div class="border-red-600 border-4 absolute left-80 top-20 p-1.5 text-lg">
  How to build a label-efficient ML System?
</div>
</v-click>

<v-click>
<div class="border-red-600 border-4 absolute h-13 w-27 left-129 top-37 "></div>
<div class="border-red-600 border-4 absolute h-13 w-27 left-163 top-82 "></div>
</v-click>

<style>
</style>

<!--
Here, I show a typical life cycle of an ML application, where we have five different phases, namely, data collection, data annotation, model training, model evaluation, and model deployment. For many real-world ML applications, data annotation can be a big headache, especially for the domain where a high level of expertise is needed to label the data, such as medical, law, and finance. (click) So, the motivation question of this project is how can we build a label-efficient Machine Learning System? More specifically, how can we build a data annotation tool that facilitates the data annotation process? In this project, we will utilize the recent advancements in two areas of machine learning to build a data annotation tool for the named entity recognition task. (click) One is called active learning, the other is transfer learning
-->

---

# Why AL + TL ?

<table style="width:80% ;margin-left:auto;margin-right:auto;">
  <caption class="p-4 text-lg">
  Table 1: Comparison of Active Learning and Transfer Learning 
  </caption>
  <tr>
    <th> </th>
    <th> <div class="title"> Active Learning (AL) </div> </th>
    <th> <div class="title"> Transfer Learning (TL) </div> </th>
  </tr>
  <tr>
    <td> Goal </td>
    <td> Select most informative example </td>
    <td> Reuse knowlege learned elsewhere</td>
  </tr>
  <tr>
    <td> Effect</td>
    <td> <emp> Reduce number of labeled data </emp> </td>
    <td> <emy>Reduce model training time </emy> <br><br> <emp> Reduce number of labeled data </emp> </td>
  </tr>
  <tr>
    <td> Target </td>
    <td> <emp> Data </emp> </td>
    <td> <emp> Model </emp> </td>
  </tr>
</table>

<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}

th, td {
  text-align: center;
}

emp {
  background-color: rgba(239, 68, 68, 0.3);
  font-size: 1.125rem;
  line-height: 1.75rem;
  padding: 0.25rem;
}
emy {
  background-color: rgba(252, 211, 77, 0.3);
  font-size: 1.125rem;
  line-height: 1.75rem;
  padding: 0.25rem;
}

.title {
  font-size: 1.25em;
  font-weight: bolder; 
}
</style>

<!--
So, why do we want to combine these two techniques in the first place?

The goal of active learning is to identify the most informative examples from a pool of unlabeled data. By labeling these more informative data, we believe that we can train a model with a fewer number of labeled data.

In contrast, transfer learning aims to reuse the knowledge learned from a similar domain or a different domain. It can both reduce the model training time and the number of labeled data. One typical way that uses transfer learning is to train a model from a pre-trained model rather than from scratch. 

As you can see, these two techniques both can reduce the number of labeled points. However, they do not overlap with each other. Because, active learning focus on the data side, while transfer learning focuses on the model side. That is to say, we can take advantage of both of them!

The rest of the presentation aims to give a high-level overview of how we combine those two techniques to build a label-efficient Machine Learning system. Specifically, we will build active learning and transfer learning-powered data annotation tool for the named entity recognition task. Now, let's see how it works and then have a closer look at the technical details behind it.
-->

---

# Live Demo - [DANER](https://github.com/VectorInstitute/DANER)

<div class="wrapper grid">
  <div class="items-center"> 
    <div class="vc">
    <img src="final_nerservice.jpg" class="max-w-xl m-auto" width="300">
    <figcaption class="text-center pt-4"> Figure 2: NER Inference Service
</figcaption>
    </div>
  </div>

  <div>
  <img src="final_neranno.jpg" class="max-w-xl m-auto" width="350">
  <figcaption class="text-center pt-4"> Figure 3: NER Annotation Service</figcaption>
</div>
</div>

<style>
.wrapper {
  grid-template-columns: 1.5fr 2fr;
}

.vc {
  position: relative;
  top: 50%;
  transform: translateY(-50%);
}

</style>

<!--
Now, let me introduce the following two services to you guys, one is called the NER inference service. It will give you a sense of what the named entity recognition task is. Basically, this service can take in some raw text and tell you which tokens are entities. To build this inference service, we need to train a model on a labeled dataset. So, to facilitate the data annotation process, we build the second service, the data annotation service.

(open the frontend on the left and backend on the right)

That is a quick glance at how the front end of our data annotation tool looks like. Let's get back to the slides
-->

---

# How to implement?

<div class="wrapper grid gap-4">
  <div class="pt-2 items-center"> 
    <div class="vc">
    <img src="annotation_tool.jpg" class="max-w-xl m-auto" width="425">
    <figcaption class="text-center pt-4"> Figure 4: Data Annotation Process
</figcaption>
    </div>
  </div>

  <div class="pt-2">
  <img src="annotation_service.jpg" class="max-w-xl m-auto" width="375">
  <figcaption class="text-center pt-4"> Figure 5: Annotation Service Structure Overview</figcaption>
</div>
</div>

<v-click>
<div class="border-red-600 border-4 absolute h-7 w-27 left-19 top-64 "></div>
</v-click>

<v-click>
<div class="border-red-600 border-4 absolute h-7 w-27 left-54 top-46 "></div>
</v-click>

<style>
.wrapper {
  grid-template-columns: 1.5fr 2fr;
}

.vc {
  position: relative;
  top: 50%;
  transform: translateY(-50%);
}

</style>

<!--
Let's see how we implement the data annotation tool.  Figure 4 shows the data annotation process we just did in the live demo. (click) In the beginning, the frontend receives a set of unlabeled data and we label it and send it to the backend server. (click) On the backend, the model is trained on all the labeled data using transfer learning. Then, we use the trained model to get the prediction on all the unlabeled data and use an active learning algorithm to compute the score for each data point. After we sort the labeled data according to the score, we send the top most informative unlabeled data to the frontend for labeling.

The right figure illustrates the overall structure of the annotation service. We implement the backend using the FLASK framework and implement the frontend in Vue.js. The most important component in the backend is the active learning engine, which is built using PyTorch, BAAL, and hugging face transformer library. The AL engine has two main functions, update model and update candidate. When you call the update model function, the AL engine will train a model using all the labeled data using transfer learning. And when you call the update candidate function, the active learning algorithm will come in to compute the score for each data point and sort them according to the score.
-->

---

# Update Model - Transfer Learning

```python {all|1|2-8|9-10|12-17|19|all}
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
training_args = TrainingArguments(
    max_steps=MAX_STEPS,             # total number of training steps per AL step
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

trainer = Trainer(
    model=hf_model,                  # the instantiated 🤗 Transformers model to be trained
    tokenizer=tokenizer,             # the tokenizer that is compatible with model
    args=training_args,              # training arguments, defined above
    train_dataset=active_set,        # AL dataset
)

trainer.train()
```

<!--
Thanks to the open source community like hugging face, we have access to a wide range of different model checkpoints.
-->

---

# Update Candidate - Active Learning

- Maximum Normalized Log-Probability (MNLP)
  - Intuition: select the most uncertain points.
  - How to represent the uncertainty?
    - Let $\mathbf{x}_{i}$ be the $i^{th}$ data point with $n$ tokens.
    - The probability of model current prediction:
      $\max _{y_{1}, \ldots, y_{n}} \mathbb{P}\left[y_{1}, \ldots, y_{n} \mid\mathbf{x}_{i} \right]$
    - The normalized log probability of model current prediction on $\mathbf{x}_{i}$ is computed as,
      $$ MNLP(\mathbf{x}_{i}) = \max _{y*{1}, \ldots, y*{n}} \frac{1}{n} \sum*{j=1}^{n} \log \mathbb{P}\left[y*{j} \mid \mathbf{x}\_{i} \right]$$
  - AL Data Candidate $\left\{ \mathbf{x} \right\} = \argmin _{\mathbf{x}_{i}}MNLP(\mathbf{x}_{i})$

<v-click>
<div class="border-red-600 border-4 absolute left-117 top-55 h-8 w-66 text-lg"></div>
</v-click>

<v-click>
<div class="border-red-600 border-4 absolute left-66 top-72 h-16 w-130 text-lg"></div>
</v-click>

<v-click>
<div class="border-red-600 border-4 absolute left-143 top-95 text-lg">
  Select data with small MNLP score!
</div>
<div class="border-red-600 border-4 absolute left-130 top-105 text-lg">
  Small score = Low Confidence = High Uncertainty
</div>
</v-click>


---

# Limitation and Next Step

<div class="wrapper grid gap-4">
  <div class="pt-2">
    <ul>
      <li>Active Learning Performance Gap
        <ul>
          <li> Dataset? Architecture? </li>  
          <li> Transfer Learning? </li>  
          <li> More robust AL Algorithm is needed </li>  
        </ul>
      </li>
      <li> Noisy Human Label 
        <ul>
          <li> Human label may not be reliable </li>  
          User 1
          <img src="disagree_1.png">
          User 2
          <img src="disagree_2.png">
          <li>Data Quality Control</li>
        </ul>
      </li>
    </ul>
  </div>
  <v-click>
  <div class="items-center"> 
    <div class="vc">
    <img src="al_performance.jpg" class="max-w-xl m-auto" width="400">
    <figcaption class="text-center pt-4"> 
    Figure 6: AL Performance from Literature
    <div class="text-xs inline">[1]</div>
    </figcaption>
    </div>
  </div>
  <div name="footnote1" class="text-xs absolute left-127 top-113 ">
  <a href="https://arxiv.org/abs/1707.05928">[1] Deep Active Learning for Named Entity Recognition (ICLR2018)</a>
  </div>
  </v-click>
</div>

<style>
.wrapper {
  grid-template-columns: 1.5fr 2fr;
}

.vc {
  position: relative;
  top: 50%;
  transform: translateY(-50%);
}

a {
  
}

#footnote1 {
  position: relative;
  transform: translateY(-50%);
}
</style>

<!--
That's the overview of the system and this system is not perfect. Let's look at some limitations. The first thing I notice is that there is a significant performance gap between real life and literature. (click) The right figure is from a Paper called deep AL for NER. We can see that all active learning algorithms achieve better performance than random baseline. Because, when we fix x -the percent of words annotated, all active learning algorithms achieve a higher F1 score. However, in my experiment, I find that active learning achieves similar performance to the random baseline. This may due to differences in experimental setups. For example, the dataset and architecture are different. And we use transfer learning rather than train model from scratch. However, no matter what the root cause of such a performance gap, it suggests that we need a more robust active learning algorithm.

The second thing I notice is that human labels may not be reliable and the noisy label can affect the model performance a lot. Here, I give a toy example where two users give two different annotations to a single data point. The first user gives the wrong Date label and the second user does not recognize that Geoffrey Hinton is a person. Both of the annotations are not perfectly correct. So, we really need a data quality control mechanism to resolve the disagreement. That would be an interesting feature for the next step.
-->

---

# Take away

<div class="pt-4">
<img src="take_away.jpg" class="m-auto" width="850">
<figcaption class="text-center pt-4"> 
Figure 7: Internship Take Away
</figcaption>
</div>

<!--
Finally, let me spend a little bit of time talking about myself. Before this internship, I just finished my undergraduate study at UofT Engineering Science. And after this internship, I will start my Ph.D. with Jimmy. So, this summer internship experience is a bridge between my undergrad study and graduate study. In my undergrad study, my research focuses on active learning, the goal of my research is to develop label efficient and computationally efficient active learning algorithms for computer vision tasks. However, I did not think too much about how we can apply this algorithm to real-world problems. This applied machine learning internship gives me the opportunity to create a product that uses my knowledge. Through this experience, I expand my skill sets and become a full-stack machine learning researcher. I believe the skills I learned this summer could help my research life in the future. Last but not least, I got a different mindset and perspective towards machine learning. Previously, I care most about are the model and algorithm. Now, I think more systematically about the data and the real-world problem. I think I have become much clearer about what types of research I want to do in my Ph.D. life.
-->
