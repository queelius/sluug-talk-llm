from graphviz import Digraph

class TreeNode:
    def __init__(self, operation, children=None, values=None):
        self.operation = operation
        self.children = children if children is not None else []
        self.values = values  # For leaf nodes

    def __repr__(self):
        if self.values is not None:
            return str(self.values)
        else:
            return f"{self.operation}"

# Leaf nodes
leaf1 = TreeNode(None, values=[4, 3])
leaf2 = TreeNode(None, values=[3, 5, 2])

# Operations
sum_node = TreeNode("sum", [leaf1])
max_node = TreeNode("max", [leaf2])
srt_node = TreeNode("srt", [sum_node, max_node])

# Hardcoded result node, assuming the operations yield [7, 7]
hardcoded_result = TreeNode("[6, 7]")

# "=" node
equals_node = TreeNode("=", [srt_node, hardcoded_result])

def visualize_tree(node, graph=None, parent=None):
    if graph is None:
        graph = Digraph()
    graph.node(name=repr(node), label=repr(node))

    if parent:
        graph.edge(repr(parent), repr(node))
    
    for child in node.children:
        visualize_tree(child, graph, node)

    return graph

# Visualize the tree
tree_graph = visualize_tree(equals_node)
tree_graph.render('simplified_expression_tree', format='png', cleanup=True)
print("Simplified tree visualization with '=' node generated as 'simplified_expression_tree.png'")
