# Dataset

The project loads `stanfordnlp/imdb` from Hugging Face when `python train.py`
runs. No dataset copy is committed to this repository. The default training
sample is 10,000 rows from the labeled training split; use `--sample-size 0`
to train on all available labeled rows.

The source split has only **negative (0)** and **positive (1)** labels. It does
not contain neutral examples; this project does not fabricate neutral labels.
Rows with missing review/label values and duplicate cleaned reviews are
identified and excluded before training.
