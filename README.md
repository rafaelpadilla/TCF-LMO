# Anomaly Detection Network
Anomaly Detection Network is a network made with dedicated modules to process frames and identify the presence of anomalies.

* Overview
* Dissimilarity Module (DM)
* Differential Morphology Module (MM)
* Temporal Consistency Module (TCM)
* Classification Module (CM)

## Training

Use the command below to train the network:

`python train.py --fold 1 --net DM_MM_TCM_CM`

Use the argument `--fold` to select which fold to train (options: `--fold 1`, `--fold 2`, `--fold 3`, `--fold 4`, `--fold 5`, `--fold 6`, `--fold 7`, `--fold 8`, `--fold 9`).

Use the argument `--net` to train the pipeline with modules placed in different orders (options: `--net  DM_MM_TCM_CM`, `--net DM_TCM_MM_CM`).

Check all possible arguments with the command `python train.py --help`

