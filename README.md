# Unification-based Explanation Reconstruction

This is repository reproduces the EACL 2021 paper [Unification-based Reconstruction of Multi-hop Explanations for Science Questions](https://arxiv.org/abs/2004.00061). 


The data (`data/`, `lemmatization-en.txt`) and the evaluation code (`evaluate.py`) are taken from its original [repository](https://github.com/ai-systems/unification_reconstruction_explanations), while I rewrite the experiment flow (`experiment.py`) and the code for the knowledge bases (`lib/knowledge_base.py`) so that they can be adapted to other tasks more intuitively.


<img src="https://i.imgur.com/k4GMLAM.png" width="600" >

#### Run the experiment

This will generate a txt file `prediction.txt` that ranks the relevant facts for each question in the dev set. 

```python
python experiment.py
```

#### Evaluation

This will evaluate your prediction result, which sould produce a MAP value 0.5455.

```python
python evaluate.py --gold=./data/questions/dev.tsv prediction.txt
```


## References
+ [Unification-based Reconstruction of Multi-hop Explanations for Science Questions](https://arxiv.org/abs/2004.00061)
+ [The original github along with the paper](https://github.com/ai-systems/unification_reconstruction_explanations)
