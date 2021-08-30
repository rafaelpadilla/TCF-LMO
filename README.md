# Anomaly Detection Network
Anomaly Detection Network is a network made with dedicated modules to process frames and identify the presence of anomalies.

* Overview
  * Dissimilarity Module (DM)
  * Differential Morphology Module (MM)
  * Temporal Consistency Module (TCM)
  * Classification Module (CM)
* Requirements
* Training
* Testing 
* Results


## Overview
The anomaly detection network consists of five modules as illustrated below:

<!--- IOU --->
<p align="center">
<img src="https://github.com/rafaelpadilla/differentiable-anomaly-detection-pipeline/blob/main/aux_imgs/pipeline_outputs_v3.png?raw=true" align="center"/></p>

Where: 

(a) aligned reference frame  
(b) target frame (where the anomaly is present)  
(c) output of the dissimilarity module (DM)  
(d) output of the temporal consistency module (TCM)  
(e) eroded version of (d) computed by differentiable morphology module (MM)  
(f) eroded version of (e) computed by differentiable morphology module (MM)  

## Requirements

If you use conda/anaconda, use the file environment.yml to install the needed packages to run the network:
`conda env create -f environment.yml`

## Training

Use the command below to train the network:

`python train.py --fold 1 --net DM_MM_TCM_CM`

Use the argument `--fold` to select which fold to train (options: `--fold 1`, `--fold 2`, `--fold 3`, `--fold 4`, `--fold 5`, `--fold 6`, `--fold 7`, `--fold 8`, `--fold 9`).

Use the argument `--net` to train the pipeline with modules placed in different orders (options: `--net  DM_MM_TCM_CM`, `--net DM_TCM_MM_CM`).

Check all possible arguments with the command `python train.py --help`

## Testing

You can download the **pre-trained** models from here:
https://drive.google.com/drive/folders/1-ExzI5REX-Hht-SixjsLJHPZ1S1zpR5y?usp=sharing
