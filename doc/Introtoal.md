# Slides_pitch

Hi everyone, I am Yongchao. Today, I will give a short presentation on my summer project - build a customized NLP Service.

## Background 1

- To give you guys a little background, this project is motivated by two observations.
  - One observation is that for most real-world machine learning applications, we repeat the process of collecting data and training the model. As we keep getting more data, our model achieves better performance. However, anntating a high quality dataset could be a headache for many practioners. Moreover, there is a diminishing return of getting more data. One possible reason is that the model has already grasped most of the knowledge within the dataset. Therefore, we probably need a better way to collect data using active learning. 
  - Another observation is that in deep learning, people are getting larger model every year. Up until now, the largest NLP model is the Wudao 2.0 system trained by AI researchers in China, which is just released yesterday. They train a 1.75 trillion parameters model which is 10 times larger than the GPT3 model and achieve the state-of-the-art result on many benchmarks datasets. As you may know, many such large-scale models, like GPT-3 and BERT are found to geralize well to different dataset and are capabale of few shot learning, which offers the opportunity to apply transfer learning. 

## Background 2

- So, how the vector institute could help our sponsors, especially for those with a relatively small machine learning bussiness.  Maybe we can provide them a machine learning workflow which utilizes the recent advancement in active learning and transfer learning. Basically, we want to give them the ability to build their own NLP service from scratch using the Data Annotation Tool and the NLP service provided by Vector.

## Example: Customized NER Service

- Let’s see a concrete example of what they potentially can do. Let’s say, they want to build a customized Name Entity Recognition System. Basically, at the start of the bussiness, they want to annotate their own dataset using our active learning powered data annotation tool. Then, they can finetune their own model based on the large-scale pretrained model. Finally, they can deploy their own model as their customized NLP service.

## What’s missing

- As you can see from the previous slide, people have already built annotation tool and NER service. So, What are missing from the current enterprise solution?
- For the data annotation tool, we find that most tools do not use active learning and those who support active learning are generally not open source. Moreover, the customers have no control over the underlying AL algorithm. For example, they can not choose Advanced AL algorithm with better Uncertainty and Diversity Trade-off, they can not tune the Auto-Suggestion Strength and the system does not consider the User Preferences.
- And for the Name entity recognition systems
  - People may prefer a customized solution tailored to their field rather than a general solution. They may want a Customized Model + Customized Entity Support, especially for the field like Law, Medical and Finance.

## Deliverables

- To sumup, to complete this summer project of building a customized NLP service from scratch, I will build two Web-Applications, namely, an active learning powered Annotation Tool and a Customized NER Interface. Besides, I will provide a Tutorial and Demo on how to build a customized NER service based on active learning and transfer learning. 
- That’s the end of my presentation. Thank you!

# Intro to Active Learning

### Title Page

Hi everyone, today, I will give a short presentation on active learning. The goal of this presentation is to give you guys a high level view of active learning and talk about the intuition behind the most commonly used active learning algorithms. 

### Outline

Here is the outline. Basically, the presentation is organized in the way to address the following three questions. What is active learning? What kind of examples are most informative? How does AL fit into the ML workflow?

### What is active Machine Learning?

So, What is active Machine Learning and why we need it? As you may know, in Machine learning, you can often get state-of-the-art results with good data and very simple algorithms. However, you rarely get state-of-the-art results with a cutting-edge algorithm built on bad data. Besides, annotating a high-quality dataset is costly, especially for the applications that requires a high level of expertise, such as medical imaging, medical NLP.

Active learning (AL) is an effective way to reduce the labeling costs while retaining the model performance. Compared with the traditional passive machine learning workflow in Figure 1, where a large dataset is labeled all at once. Figure 2 shows that active learning algorithms iteratively build up a small training set by collecting labels only on the informative input data. When we reach some stopping criterion, for example, we use up our labeling budget, a final model is trained on all the labeled data. Now, Let‘s have a close look into the active learner.

### Active Learner \& Label Efficiency

Figure 3 shows the active learner's job at each query step. Basically, we train a model using all the labeled data provided by an oracle and use the trained model to extract the feature for all the unlabeled data in the data pool. Then, an active learning algorithm is used to identify the most informative examples and send them to an oracle for labeling. After that, a next iteration begins. Through this interative process, the active learning algorithm is able to build a high-quality dataset such that the model trained on it can achieve as high test accuracy as possible. So, intuitively, what active learning algorithm cares most about is the label efficiency.

### Typical Results

Here, I show a typical figure in the active learning literatures. The x-axis is the number of annotation and the y-axis is the model’s performance. For a classification task, it is usually the test accuracy. Basically, the figure compares the model performance with and without active learning. There are usually two types of argument to show the effectiveness of active learning. The first is Given a target test accuracy, the active learning algorithm reduces the labeling cost by x%. The second is Given a fixed labeling budget, the active learning algorithm improves the model performance by x%. These two arguments are essentially the same. The first one interpret the figure when we fix the y-value. While the second one fix the x value. 

### What kind of examples are most informative?

So, how to achieve high label efficiency? Basically, we need to ask the question what kind of examples are most informative? Let’s consider a standard binary classification task. In Figure 5, we are given the current model decision boundary and all the labeled data belong to two different classes represented by blue and red circles. The goal of active learning algorithm is to identify the most informative unlabeled data represented by the white circle. Most Active learning algorithms follow the two principles: uncertainty sampling and diversity sampling. As shown in Figure 6, the goal of uncertainty sampling is to query new examples on which the model is most uncertain. Typically, they are the examples close the the current model's decision boundary because when we include these points to our training sets, the decision boundary will be very likely to change. In constrast, diverisity sampling aims to select a diverse subset of points that can act as a surrogate for the full dataset. Typically, the selected examples spread out the whole data space, as is shown in Figure 7. 

### Uncertainty Sampling

Let's look at some concrete examples. We will see that different machine learning models have different interpretations of uncertainty.

- The Top Left figure shows the decision boundary from a Support Vector Machine. Support Vector Machine is a discriminative learner which attempts to find a way to optimally divide the data and maximize the width of the decision boundary. So, the most uncertain points in SVM are the points near decision boundary because the model is likely to make a mistake there. To select uncertain points, we can calcuate each point’s distance to the decision boundary analytically, and select points with the smallest distance to the decision boundary.
- The Top Right figure shows a potential Bayesian Model. It is a Generative Supervised Learning model which means that is trying to model the distribution of each label, rather than model the boundary. Basically, we can know the probability of each example being a particular class. So, the uncertain points in this case can be the points who have a similar probability across different classes.
- The Bottom Left figure shows a Decision Tree model. In this example, the data space is splitted into four regions with the top left and right region be the class A, and the bottom left and right region be the class B. The confidence of decision tree is defined by the percent of a label in the final region. For example, in the bottom left region there is 1 Label A and 3 Label B, so a prediction in that region would be 25% confidence in Label A and 75% confidence in Label B. However, it may not be appropriate to define the uncertain example based on this probability. Since This probability is not reliable because Decision Trees are very sensitive to how far you let them divide. They could end up with just one items, which has a confidence of 100%.
- To overcome this, we can use an ensemble of Decision Trees, which is also known as “Random Forest”, as shown in the bottom right figure. Bascially, in random forest, different trees are trained on different subsets of the data and/or features. The confidence in a label can simply be the percent of times an item was predicted to be a certain class by each of the ensemble member, or the average confidence across all predictors. Then, we can define the uncertain example based on this confidence. Intuitively, the uncertain points are the ones with the most disagreement among the ensemble members.

### Diversity Sampling

The goal of diversity sampling is to select a diverse subset of points that can act as a surrogate for the full dataset. Based on the feature of the data points, a straight forward diversity sampling idea is to use clustering-based sampling. Basically, we can divide our data into a large number of clusters based on any clustering alogrithm you know, say like K-means, spectral clustering, GMM and whatever. And then, sample evenly from each cluster. you can also have various ways to sample a data from the cluster. For example, you can use random sampling or sample the data closest to the centroids. 

The other type of idea is called representative sampling. Basically, We want to find the unlabeled data that looks most like the data in the application domain where we are deploying the model. This method is rarely used in isolation. Most of time, it will combine with the uncertainty sampling approach. And the performance of it really depends on the representation we learn and the metric we are using to measure the similarity.

### Hybrid Method

Besides, there are also some hybrid methods that trade-off between uncertainty and diversity. Figure 12 shows that the examples selected by hybrid method are all close to the decision boundary, but not so close to each other. Figure 13 is a typical hybrid method, where the active learning algorithm first identify a set of  examples that models are uncertain about using some uncertainty sampling method. Then, the algorithm run a clustering algorithm on the selected examples. The final example for labeling are selected from each cluster to ensure the diversity.

### Active Learning for Neural Networks

Now, let us see, how we can implement uncertainty and diversity sampling in the context of neural networks. I guess most people here have some basic understanding of neural network. In neural networks, people have various way to define the uncertainty. Based on neural network’s prediction, the uncertain example can be the point which has the least confidence on its prediction or the points whose prediction has the maximum entropy. From a model perspective, the uncertain example can be the points that have the largest gradient norm, which will change the model the most. Or it can be the point that reduce the model’s predictive variance the most. We can also use the ensemble model to represent the uncertainty, the most uncertain examples can be the points which the ensemble models disagree the most. In terms of diversity sampling, people always formulate the problem as a coreset construction, either based on model prediction or hidden layer features. 

### Pros and Cons of different AL algorithm

Here, we summarize the Pros and Cons of different active learning algorithms. For Uncertainty sampling methods, they are usually simple to implement, but since they do not consider the data similarity, they are likely to select redundant data when querying a batch of training examples at once. In contrast, diversity sampling has good sample diversity as the selected data spread out the whole data space. However, they tend to select easy examples or outliers that do not improve the model performance. Therefore, It seems that hybrid methods are the way to go because they consider both uncertainty and diversity. However, these algorithms have some issues of scalability.

### Active Learning - An interative process

To sum up, active learning is an interative process which aims to build a high-quality dataset. Here is an example with two query steps. Basically, we are given a model and pool of unlabeled dataset. We first apply Active Learning to sample items that require a human label to create additional training items. Then, we retrain the model with the new training items, resulting in a new decision boundary. Then, we apply the active learning again to select a new set of items that require a human label. After that, we retrain the model again, and repeat the process to keep getting a more accurate model.

### How does AL fit into the ML workflow?

- Now, we have a basic understanding of what active learning is and how a typical active learning process looks like. Let’s see what’s the role of active learning in a traditional machine learning workflow. We can gain some intuition from the knowledge quadrant for Machine learning.
- The Known Knowns represents what your Machine Learning model can confidently and accurately do today. This is your model in its current state. The Known Unknowns is what your Machine Learning model cannot confidently do today. You can apply Uncertainty Sampling to these items. The Unknown Knowns can be the knowledge within pre-trained models that can be adapted to your task. We can apply Transfer learning to use this knowledge. Finally, the Unknown Unknowns represents the gaps in your Machine Learning model where it is blind today. and we can apply Diversity Sampling to these items.
- The columns and rows are also meaningful, with the rows capturing knowledge about your model in its current state, and the columns capturing the type of solutions needed:
  - The top row captures your model’s knowledge.
  - The bottom row captures knowledge outside of your model.
  - The left column can be addressed by the right machine learning algorithms.
  - The right column can be addressed by human interaction, like active learning. So, the AL is just a way to involve the human in the Machine learning workflow. And it is focus on data side rather than model or algorithm side. It is tangential to the other techniques like model architecture, training strategy, transfer learning.

### Human-in-the-Loop Machine Learning

Now, Let’s include active learning and transfer learning to our machine learning workflow. Here is a visualization of Human-in-the-Loop Machine Learning workflow. Note that, In the real world machine learning application, there is always a iterative process where you need to collect and label data, retrain the model, and deploy the model. However, most of the real world ML application may not have an active learning stage. Instead, they just label a random subset of the collected data, which seems not label efficient. 

That’s the end of my presentation. Thank you for your listening.