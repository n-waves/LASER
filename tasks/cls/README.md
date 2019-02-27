# LASER: application to CLS

This codes shows how to use the multilingual sentence embedding for
cross-lingual document classification, using the CLS corpus.

We train a  document classifier on one language (e.g. English) and apply it then
to several other languages without using any resource of that language
(German, French, Japanese)

## Results

We use an MLP classifier with two hidden layers and Adam optimization.

|  Train language      |    en  |   de  |   fr |
|----------------------|--------|-------|------|
| en:                  |  84.55 | 84.15 | 83.90|
| de:                  |  82.60 | 85.20 | 83.05|
| fr:                  |  77.20 | 82.95 | 84.85|

All numbers are accuracies on the test set, `books` subset.
