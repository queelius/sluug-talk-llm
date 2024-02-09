## Introduction

This is a presentation on the topic of large language models (LLMs) and their implications for the future of AI and society. The presentation is intended for a general audience, and will cover the following topics:

- A brief history of AI and machine learning
- The era of deep learning
- The era of self-supervised learning using giant neural networks
- The idea of prediction = compression = intelligence
- The role of LLMs in approximating this idea
- The importance of local LLMs and the alignment problem
- The future of LLMs and their implications for society



## Background
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

## Large Langage Models

Pretrained LLMs are a type of neural network that has been trained on a large corpus of text. They are trained using a technique called self-supervised learning, which is a type of unsupervised learning. The idea is to train the model to predict the next word in a sequence of words. Notice what we said: it is
learning to predict the future based on the past. Recall that prediction is the raw material of intelligence. Whether it's a stochastic parrot or a transformer, the model is learning to predict the future.  Consider this example of a murder mystery novel and near the end of the book we have the
big reveal: "...based on all of the evidence, I have determined that the killer is _".

In order to predict who the killer is, you must be able to reason about all of the evidence and hints
in the story laid out by the author.

### Example Output

Let's explore 

## Alignment Problem 

The LLM is a powerful *raw* induction (prediction) engine. It is not the only piece, but it is an important piece. It must also be able to *search* over possible futures. This is also a hard problem, and
it's being worked on, but there are already examples where when we use an LLM along with some guided search process (see Alpha Code 2 and Tree-of-Thoughts), we can get much better goal-directed problem solving. At this point, we're not trying to predict the next word, but to perform certain tasks, like solving math problems.

We can *bias* the LLM to produce outputs that we find more useful. We don't always just want it to
predict the next word -- that is a good start, but remember, we want it to do things for us. We
can imagine that the LLM has an action policy: given the task, what should it do -- at each stop,
which output should it generate? More broadly, this is called the **alignment problem**. We want the
LLM to be aligned with our goals. In RL, the policy is the function that maps states to actions. In
the case of the AR-LLM transformer, the state is the current context, and the action is the next word.

There are a few ways we can bias the LLM, or rather, generate a policy we find more useful for the task
we want it to do, besides just doing a raw search over outputs and having some kind of way to score
them to choose the best one:

1. The first is to use *prompting strategies*. The raw pretrained LLM has usually just learned to
predict a bunch of text its seen, often from random sources on the internet. So, when you give it a
propt "To make a cake, you need to", it will generate a list of ingredients and instructions that
*usually* follows in the training data. This is a form of prompting. You are conditionally sampling
from the LLM. You are saying, "given this prompt, what is the most likely next word?" This is a form
of biasing the LLM. You are telling it to generate text that is more likely to be useful to you.
However, this is difficult to do and often requires a lot of trial and error. We want to *align* it
to do what we want. However, there are a few *general* prompting strategies that work well. A notable example is chain-of-thought (CoT) prompting, where we simply write something like "Q: <task>. Let's think step by step." and this causes the LLM to conditionally sample from the part of the distribution in which people have spelled out in detail how to do the task. This is valuable in two independent ways: first, when the steps are laid out in detail, generally the data is of higher quality, and second, autoregressive models help the model to connect the inputs and outputs through intermediate steps so that the LLM's neural network has a more direct path to the output. There are fewer latent (unseen) variables for the LLM to model and average over.

2. The second approach is to to fine-tune. This is a supervised learning process where we  the pretrained model a lot of examples of the task we want it to do, and it learns to predict the correct "action" (output). This comes in two popular flavors:

    (1) Instead of training on general text sources, train on high quality sources that are more
    relevant to the task. For example, if you want to train a chatbot, you might train it on
    conversations between humans. This is called *data selection*. You are selecting the data that
    the model is trained on to be more relevant to the task you want it to do. This is a form of 
    biasing the LLM. You are telling it to generate text that is more likely to be useful to you.

    (2) Instead of next-token prediction, train the model to predict the correct output. This is
    called *instruction tuning*. You are telling the model to generate text that is more likely to be
    A popular example is RLHF, which is a form of fine-tuning that uses reinforcement learning to
    train the model to generate text that is more likely to be useful to you. RLHF is also important
    because often we do not really know how to score the outputs -- we don't really know what we want --
    but we can often compare two different outputs and say which one is better. This is called
    relative ranking. RLHF is a form of relative ranking.

In general, reinforcement learning is a way to *go beyond* the training data, and to learn to
do things that humans can't do. It is similiar to self-play in games, where the model learns to
play against itself and improve. Often, this is a good approach when the desired output may not be
known, but once it is given, we can tell if it's good or not. For instance, most problems in
NP-hard are like this. We don't know the answer, but once we have a candidate, we can tell if it's
good or not. It's like a one-way function: it's easy to go from the output to the input, but hard
to go from the input to the output. Another example: integration. It's easy to take the derivative
of a function, but hard to integrate a function.

Many believe that scale is all we need: scale in training data (mostly synthetic data), scale in
compute (for search and training), and scale in model size. The jury is still out on this, but
it seems to be the case that scale is very important. Very few people foresaw the success of
transformers, and it seems that the main reason they work is because they are so big. This is
a very important point.

(They also have some useful indutive biases, e.g., in-context learning, which is a form of
transfer learning.)

## LLMs in Practice

We've all seen the amazing things that LLMs can do. If not, try it out. We're gonna use one now,
ChatGPT from OpenAI. It's a fine-tuned chatbot. It loves to talk! So, let's talk to it.

```
example here
```


## Local LLMs and Open Source

LLamam is an open-source LLM by Meta (formerly Facebook). Since the LLM race has been heating up,
many of the large corporations have closed off their models, and a lot of the research is being done
in private. This is a problem, because we need to be able to understand these models, and for now I think that means we need open source. So, let's first discuss safety and alignment research.

### Safety and Alignment

Once these things get very intelligent or competent, they can be used to do things we don't like. For
instance, deepfakes become much easier, and potentially worse, targetted propaganda for individuals
based on their psychological profile. They'll not just write you emails, but call you up with a human
sounding voice and engage with you in a discussion designed to manipulate you. This is a very real
concern. We need to be able to understand these models, and we need to be able to control them. We
need to be able to align them with our goals. This is a very hard problem, and it's not clear that
we can solve it. But we need to try.


### Open Source
For this reason, open source models like Llama are very important. It allows applicable research
on sufficiently capable models to continue in the open, and it doesn't put all the power in the
hands of a few large corporations. 

## Outline for presentation



 i'll go over a brief history of AI/ML, era of deep learning, era of self-supervised learning using giant NNs LLM using transformers. as prelude, talk about idea of predition = compression = intelligence, and agency is a sort of search over the prediction space and choosing actions that are predicted to lead to highest expected reward (often reward is implicit). draw from solomonoff induction, AIXI (marcus hutter). show how LLMs and sequence models are an approximation of this. then i go on a whirlwind tour of LLMs and how to use them, prompting strategies (CoT, etc), then transition to local LLMs and their importance -- show uncensored models that can produce "harmful" outputs (local LLM) and address the ALIGNMENT problem/challenge. talk fine-tuning, RLHF, synthetic data, vector stores, tool use (python eval, and any other API endpoint), agentic frameworks like autogen, coordinating agents (also autogen), langchain, LLM programming (LMQL, powerful superset of python that lets us use LLMs in extremely interesting and powerful ways). show OpenInterpreter -- i'll probabably want to keep some money in my openai account so i --can use it -- and let that transition me to the idea of the LLM-OS (karpathy discusses it). this will give me a chance to talk about the future: AGI, or IA (intelligence augmentation), super-intelligence, implications, etc. i want to say that LLMs as intelligence engines are a universal glue that understands language enough to be able to interface and glue together all of the systems humans use (this will probably be near the start of the presentation) -- that this is a revolution as big as the invention of the printing press.


 other random notes:

 Downloaded a pre-trained model, not fine-tuned to be instruction tuned
where they follow directions well, nor fine-tuned to be chat bots, where they
are good at holding conversations with humans.


In both of these cases, the fine-tuning process ALIGNS the model to some
objective. Fine-tuning can just be supervised tasks where the training data
is much more specialized, like showing
ALL THIS DOES is predict the next token.
