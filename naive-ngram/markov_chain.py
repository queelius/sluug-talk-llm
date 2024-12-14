import random

class MarkovChain:
    def __init__(self):
        self.model = {}

    def percept(self,
                prev_tokens: list,
                next_token: str):
        """
        Add a percept to the model.

        :param prev_tokens A list of tokens representing the context of the
        percept.

        :param next_token The token that follows the context.
        """
        if prev_tokens not in self.model:
            self.model[prev_tokens] = {}

        if next_token not in self.model[prev_tokens]:
            self.model[prev_tokens][next_token] = 0

        self.model[prev_tokens][next_token] += 1

    def train(self, data: list, order: int = 1):
        """
        Train the model.

        If you want to just add a percept to the model, use the percept
        method. It allows you to add arbitrary percepts to the model, without
        making any assumptions about order or the structure of the data.

        :param data A list of lists of tokens, each list representing a sequence of
        tokens. The inner lists can have different lengths. the model will be
        trained on each sequence independently. This is a simple way to
        represent a corpus of text, where each list represents, say, a document,
        or a paragraph, or a sentence, or a line, or a word, depending on the
        granularity of the data.
        
        :param order The order of the Markov chain. We apply this order to the
        training data instead of the model itself, so that different orders
        can be used with the same model. This model represents a life-long
        learning system, which can be trained with different orders at different
        times.
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

    def distribution(self, ctx: list) -> dict:
        """
        Get the distribution of tokens that follow a given context.

        If the context is not in the model, we return the distribution of
        tokens that follow the context without the first token. This is recursively
        applied until we find a context that is in the model. This is a simple
        inductive bias that allows the model to generalize out of the distribution
        of the training data. Essentially, we assume that the most recent tokens
        are the most relevant. Clearly, this is not always true.

        :param ctx: A list of tokens representing the context.
        :return A dictionary representing the distribution of tokens that follow
        the given context. The keys are the tokens, and the values are the
        probabilities of each token following the context: Pr(token | ctx).
        """
        if ctx not in self.model:
            return self.distribution(ctx[1:])        
        
        total = sum(self.model[ctx].values())
        return self.model[ctx] | {k: v / total for k, v in self.model[ctx].items()}
    
    def predict(self, ctx: list) -> str:
        """
        Predict the next token that follows a given context.

        :param ctx: A list of tokens representing the context.
        :return The token that follows the context, according to the model.
        :raises ValueError: If no candidate is found.
        """

        d = self.distribution(ctx)
        r = random.random()
        s = 0
        for token, p in d.items():
            s += p
            if s >= r:
                return token
        raise ValueError('No candidate found.')

    def generate(self, ctx: list,
                 max_tokens: int = 100,
                 stop_token: str = '.') -> str:
        """
        Generate a sequence of tokens that follows a given context.

        :param ctx: A list of tokens representing the context.
        :param max_tokens: The maximum number of tokens to generate.
        :param stop_token: The token that stops the generation.
        :return A string representing the generated sequence of tokens.
        """
        output = ''
        for i in range(max_tokens):
            token = self.predict(ctx)
            output += token
            if token == stop_token:
                break
            ctx = ctx + token

        return output
    