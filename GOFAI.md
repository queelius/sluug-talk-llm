## Good-Old-Fashioned AI (GOFAI)

Let's discuss a bit more about GOFAI. In my talk, I briefly touched upon it,
but here I want to do a deeper dive. GOFAI is the era of symbolic reasoning. It
is the era of *writing down the rules*.

### Symbolic Logic
This era goes way back to antiquity, where formal logic was first developed. The hope was that we could encode human knowledge in a symbolic form and then apply a mechanical process to solve problems. A classic example:

1. Premise: All birds can fly.
2. Premise: A sparrow is a bird.
3. Conclusion: A sparrow can fly.

The encoded knowledge are the premises, and the mechanical process
is the logical inference that leads to the conclusion. If the premises are true, then the conclusion must be true. This is called *deductive reasoning*.

Typically, we're faced with incomplete information, so we need to reason under uncertainty:

1. Observation: Every sparrow we've seen can fly.
2. Prediction: The next sparrow we see will be able to fly.

This is called *inductive reasoning*. It is a generalization of the evidence to a conclusion. We often want to *understand* the evidence we have:

1. Rule: Birds are known to fly and have feathers.
2. Evidence: You see a small feathered flying creature.
3. Hypothesis: The creature is a bird.

### Probabilistic Reasoning

The real world is too complex to reason about with certainty, so we need to reason under uncertainty. In the earlier examples, each step had a degree of **uncertain**, and so we produced beliefs that are *likely* to be true. 

One of the best tools we have for dealing with uncertainty is probability theory. We can encode our symbolic knowledge in a probabilistic form and then apply the probability calculus to reason under uncertainty. For instance, we might have evidence that $S$ is a sparrow, and we want to know the probability that $S$ can fly. In *Bayesian reasoning*, we can use Bayes' rule to compute:

$$
p(S|F) \propto p(F|S)p(S)
$$

where:

- $p(S|F)$ is the probability that $S$ is a sparrow given that it can fly,
- $p(F|S)$ is the probability that $S$ can fly given that it is a sparrow, and
- $p(S)$ is the prior probability that $S$ is a sparrow.

It may be easy to estimate $p(F|S)$ and $p(S)$, for instance
the probability that of $F$ given $S$ is just the fraction of sparrows that can fly, and the prior probability that $S$ is a sparrow is just the fraction of birds that are sparrows. We can use these to compute the probability that $S$ is a sparrow given that it can fly, which may be more difficult to estimate and is the quantity we are interested in computing in the earlier example where we were trying to predict whether the next sparrow we see will be able to fly.

### The Challenge of Feature Engineering

In each of these examples, we decided which rules and symbols to use to represent the problem. In the early days of AI, we used this approach to automate various tasks. We encoded our knowledge in a symbolic form and then used computers to apply the rules. This is called *good-old-fashioned AI* (GOFAI). This is still predominately how we program computers today (and a large part of the way we *program* ourselves, arguably).

This approach has had a lot of success, particularly for problems we know how to represent symbolically, e.g., super-human chess-playing programs. However, the "real world" is not so easy to represent. For instance, how do we model even a simple problem like computing the probability that a cat is in a picture? This is a task most humans can do effortlessly, but we don't know *how* we do it, so we can't write down the rules for it. This is a very important point: we can't write down the rules for everything we can do.

