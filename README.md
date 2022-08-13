# NLP Constrained Sequential Inference

This Course project explores sequential inference in NLP, particularly using FSA - inspired by [A General-Purpose Algorithm for Constrained Sequential Inference](https://aclanthology.org/K19-1045/)

## Dependency Parsing
See [benji_transformer_play.py](benji_transformer_play.py) for training a dependency parser

`beam_search_matrix` is used for constrained sequential inference. Examples are in [dep_parse_plotting.ipynb](dep_parse_plotting.ipynb) and [dep_parse_beam_search_stats.ipynb](dep_parse_beam_search_stats.ipynb).

## Poem Generation
Try [PoemGenerationDemo.ipynb](PoemGenerationDemo.ipynb) for poem generation with normal constraints.

[text_generation_beam_search.ipynb](text_generation_beam_search.ipynb) for a similar thing but instead using FSA enforced constraints.


## Maximum Spanning Tree
We use the MST algorithm from Zmigrod: spanningtrees package forked from [https://github.com/rycolab/spanningtrees]

## Installation / Requirements
Most of the .ipynb have built in `!pip install ...` blocks, and are suitable for running locally or e.g. on google colab.

E.g. poem generation uses 
```python
!pip install transformers
!pip install torch
!pip install numpy
!pip install pronouncing
!pip install frozendict
!pip install sacremoses
!pip install Phyme
```
and dependency parsing uses

```angular2html
!apt install -y graphviz
!pip install graphviz torch torchtext transformers datasets conllu 
```