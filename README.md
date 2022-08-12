# NLP Constrained structural Inference

## Dependency Parsing
See [benji_transformer_play.py](benji_transformer_play.py) for training a dependency parser


## Poem Generation
See [PoemGenerationDemo.ipynb](benji_transformer_play.py) for training a dependency parser



## Minimum Spanning Tree
We use the MST algorithm from Zmigrod: here's their description

This library contains a modified implementation of the Maximum Spanning Tree (MST) algorithm proposed by Tarjan (1977) and Camerini et al. (1979).
The MST algorithm is a popular decoding algorithm for graph-based depenedency parsing.
However, dependency trees often have a constraint that only one edge may emanate from the root.
Gabow and Tarjan (1984) suggest an efficient extension to the algorithm for this which is included in this library.
The implementation of this library runs in the expected O(N^2) and is based on the
implementation of [Stanojević and Cohen](https://github.com/stanojevic/Fast-MST-Algorithm). soon.

This library also contains a simplified version of the K-Best MST algorithm
of Camerini et al. (1980) that runs in O(K N^2). Additionally, the
library contains a new algorithm for finding the K-Best dependency trees subject
to a root constraint.

A detailed description of these algorithms including proofs of correctness can be found in
["Mind the Root: Decoding Arborescences for Dependency Parsing"](https://www.aclweb.org/anthology/2020.emnlp-main.390/)
and
["On Finding the K-best Non-projective Dependency Trees"](https://arxiv.org/abs/2106.00780)
.
## Other things from MST README

## Requirements and Installation

* Python version >= 3.6

Installation:
```bash
git clone https://github.com/rycolab/spanningtree
cd spanningtree
pip install -e .
```


## Example
We support graph creations using numpy arrays.
We consider `0` to be the root node of the graph.
Note that the root node has no incoming edges and we do not have any self loops.
```python
import numpy as np
from spanningtrees.mst import MST
from spanningtrees.kbest import KBest

graph = np.array(([
        [ 0., 71., 52., 67.],
        [ 0.,  0., 32., 46.],
        [ 0.,  2.,  0., 79.],
        [ 0., 65., 16.,  0.]
    ]), dtype=float)
mst = MST(graph)
kbest = KBest(graph)

```
The MST and root-constrained MST can then be calculated by:
```python
print(f"Unconstrained MST: {mst.mst()}")
print(f"Constrained MST:   {mst.mst(True)}")
```
Output:
```
Unconstrained MST: [-1  0  0  2]
Constrained MST:   [-1  3  0  2]
```

The following can be instead used to find the K-Best trees
```python
print("Unconstrained K-best")
for tree in kbest.kbest():
    print(kbest.weight(tree), tree)
# Root constraint
print("Constrained K-best")
for tree in kbest.kbest(True):
    print(kbest.weight(tree), tree)
```
Output:
```
Unconstrained K-best
202.0 [-1  0  0  2]
196.0 [-1  3  0  2]
190.0 [-1  0  0  0]
184.0 [-1  3  0  0]
182.0 [-1  0  1  2]
170.0 [-1  0  1  0]
169.0 [-1  0  0  1]
164.0 [-1  3  1  0]
154.0 [-1  0  3  0]
149.0 [-1  0  1  1]
148.0 [-1  3  3  0]
133.0 [-1  0  3  1]
133.0 [-1  2  0  2]
121.0 [-1  2  0  0]
100.0 [-1  2  0  1]
85.0 [-1  2  3  0]
Constrained K-best
196.0 [-1  3  0  2]
182.0 [-1  0  1  2]
164.0 [-1  3  1  0]
149.0 [-1  0  1  1]
148.0 [-1  3  3  0]
133.0 [-1  2  0  2]
133.0 [-1  0  3  1]
100.0 [-1  2  0  1]
85.0 [-1  2  3  0]
```

## Related Work
This code repository focuses on decoding MSTs.
A useful library to use during training and learning of edges weights
can be found [here](https://github.com/rycolab/tree_expectations).

Other libraries for performing MST computations are
[Stanojević and Cohen](https://github.com/stanojevic/Fast-MST-Algorithm),
[networkx](https://networkx.github.io/documentation/stable/index.html),
and [stanza](https://stanfordnlp.github.io/stanza/).
