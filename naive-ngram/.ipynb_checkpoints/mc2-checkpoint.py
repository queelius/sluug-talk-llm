import random

class MarkovChain:
    def __init__(self):
        self.model = {}

    def percept(self, prev_tokens, next_token):
        """
        @brief add a percept to the model.

        @param prev_tokens a list of tokens representing the context of the
        percept.

        @param next_token the token that follows the context.

        """
        if prev_tokens not in self.model:
            self.model[prev_tokens] = {}

        if next_token not in self.model[prev_tokens]:
            self.model[prev_tokens][next_token] = 0

        self.model[prev_tokens][next_token] += 1

    def train(self, data, order = 1):
        """
        @brief train the model.

        @param data a list of lists of tokens, each list representing a sequence of
        tokens. the inner lists can have different lengths. the model will be
        trained on each sequence independently. this is a simple way to
        represent a corpus of text, where each list represents, say, a document,
        or a paragraph, or a sentence, or a line, or a word, depending on the
        granularity of the data.
        
        @param order the order of the Markov chain. we apply this order to the
        training data instead of the model itself, so that different orders
        can be used with the same model. this model represents a life-long
        learning system, which can be trained with different orders at different
        times.
         
        @note if you want to just add a percept to the model, use the percept
        method. it allows you to add arbitrary percepts to the model, without
        making any assumptions about order or the structure of the data.
        """
        N = len(data)
        for i in range(N):
            tokens = data[i]
            m = len(tokens)
            if i % 10000 == 0:
                print(f'{i/N*100} percent complete')
            for j in range(m):
                for k in range(0, order+1):
                    if m <= j+k:
                        break
                    self.percept(prev_tokens = tokens[j:j+k],
                                 next_token = tokens[j+k])

    def distribution(self, ctx):
        """
        @brief get the distribution of tokens that follow a given context.

        @param ctx: a list of tokens representing the context.

        @return a dictionary representing the distribution of tokens that follow
        the given context. the keys are the tokens, and the values are the
        probabilities of each token following the context: Pr(token | ctx).

        @note if the context is not in the model, we return the distribution of
        tokens that follow the context without the first token. this is a
        recursive definition, which allows us to use the model to predict
        tokens that follow any context, even if the context is not in the model.

        this represnts the only way in which the model is capable of generalizing
        out of the disribution of the training data. we assume that the most
        recent tokens are the most relevant. clearly, this is not always true.
        this represents an inductive bias of the model.
        """
        if ctx not in self.model:
            return self.distribution(ctx[1:])        
        
        # we need to normalize the counts to get a probability distribution
        total = sum(self.model[ctx].values())
        return self.model[ctx] | {k: v / total for k, v in self.model[ctx].items()}
    
    def predict(self, ctx):
        """
        @brief predict the next token that follows a given context.

        @param ctx: a list of tokens representing the context.

        @return the token that follows the context, according to the model.
        """

        d = self.distribution(ctx)
        r = random.random()
        s = 0
        for token, p in d.items():
            s += p
            if s >= r:
                return token
        raise ValueError('no candidate found')

    def generate(self, ctx, max_tokens = 100, stop_token='.'):
        output = ''
        for i in range(max_tokens):
            token = self.predict(ctx)
            output += token
            if token == stop_token:
                break
            ctx = ctx + token

        return output
    