### Machine Learning

Since we don't know how we do many things, we can't write down the rules for doing them. However, as evidenced by our brains ability to do them, we can learn to do them from experience (data).

This is the idea behind statistical learning theory. The idea is to learn a model from data. That is, a computer learns how to do things by looking at examples. So, even if we don't know know the rules, we can use an algorithm that learns the (often incomprensible) rules from data, such as learning how to recognize cats in pictures by looking at many examples of cats and non-cats.

We can categorize the *data* into two types (excluding reinforcement learning for now):

1. Supervised and unsupervised. In supervised learning, we have a set of input-output pairs, and we want to learn a function that maps inputs to outputs. This is how we normally did things in the early days of machine learning, because it doesn't need as much compute nor as much data to reach a competent level of skill at whatever narrow task we trained it to do.

2. Unsupervised learning, In unsupervised learning, we have a set of inputs, and we want to learn something about the data, such as clustering or density estimation. This is a much harder problem, because we don't have the labels to guide the learning process. We have to learn the structure of the data from the data itself.

Learning the probability distribution is the most general form of statistical learning, as it allows us to do all of the other things that we seek to do in supervised learning, too, like classification. Suppose we have data

$$
x = (x_1, x_2, x_3, \ldots, x_n)'
$$

where each $x_j$ is some kind of observation or measurement about the data. If we learn (or estimate) the probability distribution of the data, $p(x)$, we can use that to do many tasks, like:

- Classification: given an input, predict the class it belongs to. Let $y = g(x)$ be the class, where $g$ is the function that maps inputs to classes. We can use Bayes' rule to do this:

$$
p(y|x) = p(x,y)/Z
$$
where $Z$ is just a normalizing constant that ensures that the probabilities sum to 1. We can estimate $p(x,y)$ by sampling from $p(x)$ and computing the fraction of times that $g(x) = y$.
For instance, suppose $y = x_n$, then predicting the class of $x$ is just $p(x_n|x_1,\ldots,x_{n-1}) = p(x_1,\ldots,x_n)/Z$ where
$Z$ is just a normalizing constant and if $x_n$ is discrete, we
can simply sum over all possible values of $x_n$ to compute $Z$:

$$
Z = \sum_{x_n} p(x_1,\ldots,x_n)
$$

which is actually just the marginal probability $p(x_1,\ldots,x_{n-1})$. If $x_n$ is continuous, we can use the integral instead of the sum.

During this time, researchers developed many algorithms for learning from data, such as decision trees, support vector machines, neural networks, k-nearest neighbors, k-means clustering, and many others.

Even the neural network (multi-layer perceptron) was invented during this time, but it was not very popular as they were difficult to train and were computationally expensive. They were theoretically understood to be *universal function* approximators, but that assumes that we had an efficient way to train them and a sufficient amount of data to train them on. This was not the case at the time.

### The Era of Deep Learning

The biggest problem with the previous era in machine learning was
that we had to hand-engineer the features. This is a very difficult
problem: how do we *represent* the data? In the machine learning
era, we freed ourselves to use very complicated models that had the capacity to represent more complicated data, but we still had to think about how to *represent* the data.

The problem of representation is the problem of *feature engineering*. We have to decide what features to use to represent the data. This is a very difficult problem, and it is often the most important part of the machine learning process. If we don't have good features, then we can't learn a good model.

In fact, during the era of GOFAI, we not only had to decide how to represnt the data, but also we had to write down the rules that operated on the data. We no longer had to write down the rules, but we still have to decide how to represent the data.

This is the problem that deep learning solves. It learns the representation from the data. This is a very important point. We no longer have to decide how to represent the data. In fact, it's often said that deep learning is a form of representation learning. If we have the right representations, then we can learn the right model. Since the data is so complicted, normally the right model -- the right rules -- are also very complicated. This is why in deep learning, we use neural networks, which are universal function approximators.

If we give it enough data, a neural network can learn *any* function. The problem is, of course, how do we generate enough data?

### The Era of Scale

The era of scale is the era of absolutely gigantic neural networks that are trained on absolutely gigantic amounts of data. Note that in principle, we can use the same algorithms invented previously, but now on a much larger scale.

However, in order to achieve that scale, we are obliged to use
algorithms that scale with compute and data. This is the era of the transformer, and it is the era of the autoregress LLM and foundation models (multi-modal).

How do we get enough data, though? We cannot rely upon labeled data, i.e., supervised learning, as there simply isn't enough
curated data to train these models. We need to use unsupervised learning, and in particular, self-supervised learning. This is a form of unsupervised learning where we train the model to predict the next token in a sequence of tokens. This is how we learn
to estimate the probability distribution of extremely complicated data generating processes.



## On Intelligence
what is intelligence?

that's a difficult question. so, first, let's identify what's needed for intelligenc.

1. prediction. in order to be able to make intelligent choices, one must be able to predict the future to plan accordingly.

2. search. if you can predict the future well, and you can entertain multiple possible futures, then you can choose the best one. this is a form of search. this is counterfactual reasoning: if i do action A then i predict X, if i do action B then i predict Y, and i predict Y is better than X, so i do action B.

3. notice that i said "is better than", so in addition to prediction and search, we need something that places different values on different outcomes. in RL literature, this is called a reward function, but it's really just saying we prefer certain things over others.

4. finally, we need to be able to act. this is the agency part of intelligence. we need to be able to choose actions that we predict will lead to the best outcomes.

the raw material of intelligence is prediction. if you can predict the future, then you can search over possible futures, and if you can search over possible futures, then you can choose the best one, and if you can choose the best one, then you can act to make it happen. an intelligent agent also has self-knowledge, since it must be able to predict its own future states and it must also know what actions it can take, so self-awareness at this level is a byproduct of highly intelligent agents.

there is another way to look at prediction. in order to be able to predict the future, there must be
some kind of regularity in the world. if the world is completely random, then there is no way to predict the future. we can use these regularities to find smaller representations of the world. for instane,
instead of storing a precise path of a ball, we can store a simple equation that describes the path of the ball. this equation is not only a prediction of the future, but it's also a compression of its
history.

and thus, we might say that an intelligent agent is one that can compress a model of the world so that it can predict the future. this is what we mean when we say an intelligent agent needs to be able to build world models. it's what we do. in each of your heads, you have a model of the world. we each live in a simlated world inside of our heads, and we have a sort of shared hallucination of reality.

The main take-away: prediction = compression = intelligence.

There is a lot of prior work on this, but too notable examples are Ray Solomonoff's universal solomonoff induction and AIXI by Marcus Hutter. These are theoretical models of intelligence that are based on the idea of prediction = compression = intelligence. They are not practical models, but they are useful for understanding the nature of intelligence.

## Pre-Trained Large Langage Models

Pre-trained LLMs are a type of neural network that haves been trained on a large dataset.
They are trained using a technique called self-supervised learning, which is a type of unsupervised learning. The idea is to train the model to predict the next word in a sequence of words (tokens). Notice what we said: it is
learning to predict the future based on the past. Recall that prediction is the raw material of intelligence. Whether it's a stochastic parrot or a transformer, the model is learning to predict the future.

Consider this example of a murder mystery novel and near the end of the book we have the
big reveal: "...based on all of the evidence, I have determined that the killer is _".

In order to predict who the killer is, it seems you must be able to reason about all of the evidence and hints
in the story laid out by the author.

The LLM is a powerful *raw* induction (prediction) engine. To enable system 2, it
may need to be able to *search* over possible futures. This is also a hard problem, and
it's being worked on, but there are already examples where when we use an LLM along with some guided search process (see Alpha Code 2 and Tree-of-Thoughts), we can get much better goal-directed problem solving. At this point, we're not trying to predict the next word, but to perform certain tasks, like solving math problems.

### Alignment
We can *bias* the LLM to produce outputs that we find more useful. We don't always just want it to
predict the next word -- that is a good start, but remember, we want it to do things for us. We
can imagine that the LLM has an action policy: given the task, what should it do -- at each stop,
which output should it generate? More broadly, this is called the **alignment problem**. We want the
LLM to be aligned with our goals. In RL, the policy is the function that maps states to actions. In
the case of the AR-LLM transformer, the state is the current context, and the action is the next word.

There are a few ways we can bias the LLM, or rather, generate a policy we find more useful for the task
we want it to do, besides just doing a raw search over outputs and having some kind of way to score
them to choose the best one:

### Prompting Strategies

The first is to use *prompting strategies*. The raw pretrained LLM has usually just learned to
predict a bunch of text its seen, often from random sources on the internet. So, when you give it a
propt "To make a cake, you need to", it will generate a list of ingredients and instructions that
*usually* follows in the training data. This is a form of prompting. You are conditionally sampling
from the LLM. You are saying, "given this prompt, what is the most likely next word?" This is a form
of biasing the LLM. You are telling it to generate text that is more likely to be useful to you.
However, this is difficult to do and often requires a lot of trial and error. We want to *align* it
to do what we want. However, there are a few *general* prompting strategies that work well. A notable example is chain-of-thought (CoT) prompting, where we simply write something like "Q: <task>. Let's think step by step." and this causes the LLM to conditionally sample from the part of the distribution in which people have spelled out in detail how to do the task. This is valuable in two independent ways: first, when the steps are laid out in detail, generally the data is of higher quality, and second, autoregressive models help the model to connect the inputs and outputs through intermediate steps so that the LLM's neural network has a more direct path to the output. There are fewer latent (unseen) variables for the LLM to model and average over.

### Fine-Tuning

The second approach is to to fine-tune. This is a supervised learning process where we  the pretrained model a lot of examples of the task we want it to do, and it learns to predict the correct "action" (output). This comes in two popular flavors:

1. Instead of training on general text sources, train on high quality sources that are more
relevant to the task. For example, if you want to train a chatbot, you might train it on
conversations between humans. This is called *data selection*. You are selecting the data that
the model is trained on to be more relevant to the task you want it to do. This is a form of 
biasing the LLM. You are telling it to generate text that is more likely to be useful to you.

2. Instead of next-token prediction, train the model to predict the correct output. This is
called *instruction tuning*. You are telling the model to generate text that is more likely to be
A popular example is RLHF, which is a form of fine-tuning that uses reinforcement learning to
train the model to generate text that is more likely to be useful to you. RLHF is also important
because often we do not really know how to score the outputs -- we don't really know what we want --
but we can often compare two different outputs and say which one is better. This is called
relative ranking. RLHF is a form of relative ranking.

### Reinforcement Learning

Reinforcement learning is a way to fine-tune *beyond* the training data, and to learn to
do things that humans can't do, e.g., AlphaGo learned to play better Go better than any hummans
using self-play.

## Synthetic Fine-Tuning Training Data

We consider two types of data here, both of which are synthetic. The first is
data that is generated by capable models like GPT-4. The second is data that is generated by
algorithms or simulations.

#### LLM-Generated

We can use GPT-4 to generate a lot of synthetic data for the task we want it to do,
and curate it to be of high quality, or we can combine it with other mechanisms like
algorithmic or simulation data generation. We can then use this data to fine-tune
smaller models (TinyLLamas) to do the task we want it to do.

Many believe that scale is all we need: scale in training data (mostly synthetic data), scale in
compute (for search and training), and scale in model size. The jury is still out on this, but
it seems to be the case that scale is very important. Very few people foresaw the success of
transformers, and it seems that the main reason they work is because they are so big. This is
a very important point.

(They also have some useful indutive biases, e.g., in-context learning, which is a form of
transfer learning.)

#### Algorothmic

Algorithmic data generation seems like a promising approach that is currently
underexplored. The idea is to use algorithms to generate data that is useful for training models
to give them various kinds of competencies. For instance, we can use algorithms to generate data
to solve algebraic equations, or to generate data to solve physics problems.

We can also generate algorithmic data that is in a sense impractical to solve by
humans are even existing algorithms.
For instance, many problems are easy to solve in one direction, but hard to solve in the other.
We can algortihmically solve the problem in the easy direction, and then use that to train a model
to solve the problem in the hard direction.
For instance, finding antiderivatives is notoriously difficult, but finding derivatives is easy.
There are a large class of such problems that we can exploit to improve the performance of our language models.

## Tools and Agency

Humans would not be humans without tools. We are tool users. We use tools to extend our reach, our 
capabilities, and ultimately our intelligence.

One of our most powerful tools is *language*. It is a tool that allows us to communicate, to think,
to plan, to reason, to learn, and to teach. It is a source of information with a high bandwidth, and
it is a source of knowledge that is stored in a way that is easy to access and manipulate.

LLMs, trained on linguist data, are a tool that can understand and generate language. This by itself
makes them extremely powerful, but when combined with other tools, they become even more powerful.

