# LDA
LDA for arxiv data.
### Arxiv dataset
Please download and uncompress the [dataset][2] to: 
```
data/
```
### Train the Model

```
python lda.py
```
The script allows to specify the following parameters:
```
  --data-dir        Path to directory where the data is stored
  --model-dir       Path to directory where the model is stored
  --train           True for train, False for test mode
  --n_topic         Number of of topics
```
