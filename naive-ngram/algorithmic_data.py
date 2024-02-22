import random
import math

def default_args():
    return {
        'operations': [sum, math.prod, sum, min, max, sorted],
        'recurse_prob': 0.25,
        'min_child': 2,
        'max_child': 4,
        'values': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'min_depth': 0,
        'max_depth': 2,
        'debug': False
    }

def generate_data(
        samples=1, 
        stop_tok='.',
        args=default_args()):

    for k in default_args().keys():
        if k not in args:
            args[k] = default_args()[k]

    sample = []
    for _ in range(samples):
        data = generate_tree(
            args['operations'],
            args['recurse_prob'],
            args['min_child'],
            args['max_child'],
            args['values'],
            args['min_depth'],
            args['max_depth'],
            args['debug'])

        result = data['result']
        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        tok_seq = f"{data['expr']}={result}{stop_tok}"
        tok_seq.replace(' ', '')
        sample.append(tok_seq)
    return sample

def generate_tree(
        operations,
        recurse_prob,
        min_child,
        max_child,
        values,
        min_depth,
        max_depth,
        debug,
        offset=''):
    """
    @param operations a list of operations that we can apply to lists of tokens
    @param recurse_prob the probability of recursing (creating a subtree)
    @param min_child the minimum number of child nodes for each node.
    @param max_child the maximum number of child nodes for each node.
    @param values a set of values that can be sampled at leaf nodes.
    @param min_depth the minimum depth of the tree
    @param max_depth the maximum depth of the tree
    @param debug whether to print debug information
    @param offset the offset for the debug information

    @return a dictionary representing a tree, with the following keys:
    - result: the result of the evaluating the expression tree
    - data: a string representing the expression tree (unevaluated)

    @note we assume that each operation works on lists of values, e.g.,
    `sum([1, 2, 3])`. we auto-wrap if needed, so make sure your operations
    work on lists of values.

    @note the leaf values should make sense in the context of the operations.
    they should also be serializable as bytes to be used in the n-gram model.
    an operation should produce output that also makes sense in the context of
    the operations and the n-gram model.

    @note we also augment each node with a description of what the operation is
    doing to the data, so that we can generate a human-readable description of
    process of evaluating the expression, which may be useful as training data
    for larger AR models (process supervision).
    """

    if max_depth != 0 and (min_depth >= 0 or random.random() < recurse_prob):
        num_nodes = random.randint(min_child, max_child)

        childs = [generate_tree(
            operations=operations,
            recurse_prob=recurse_prob,
            min_child=min_child,
            max_child=max_child,
            values=values,
            min_depth=min_depth-1,
            max_depth=max_depth-1,
            debug=debug,
            offset = offset + '  ') for _ in range(num_nodes)]
        
        child_results = []
        for child in childs:
            if not isinstance(child['result'], list):
                child['result'] = [child['result']]
            child_results.extend(child['result'])
        op = random.choice(operations)   
        res = op(child_results)
        if not isinstance(res, list):
            res = [res]
        expr = f'{op.__name__}[{",".join([child["expr"] for child in childs])}]'
        data = {'result': res, 'expr': expr}
        if debug:
            print(f'{offset}{data=}')
        return data
    else:
        v = random.choice(values)
        data = {'result': v, 'expr': f'{v}'}
        
        if debug:
            print(f'{offset}{data=}')
        return data


