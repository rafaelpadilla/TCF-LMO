# Anomaly Detection in Moving-Camera Videos Using Deep-Learning Networks and Fully Differentiable Morphological Operations (DL-DMO)
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

<!--- scheme --->
<p align="center">
<img src="https://github.com/rafaelpadilla/differentiable-anomaly-detection-pipeline/blob/main/aux_imgs/pipeline_outputs.png?raw=true" align="center"/></p>

Where: 

(a) aligned reference frame  
(b) target frame (where the anomaly is present)  
(c) output of the dissimilarity module (DM)  
(d) output of the temporal consistency module (TCM)  
(e) eroded version of (d) computed by differentiable morphology module (MM)  
(f) eroded version of (e) computed by differentiable morphology module (MM)  


### Dissimilarity Module (DM)
<details>
<summary>Click to expand</summary>

<img src="https://github.com/rafaelpadilla/differentiable-anomaly-detection-pipeline/blob/main/aux_imgs/pipeline_DM.png?raw=true" align="center"/></p>
</details>

### Differential Morphology Module (MM)
<details>
<summary>Click to expand</summary>

<img src="https://github.com/rafaelpadilla/differentiable-anomaly-detection-pipeline/blob/main/aux_imgs/pipeline_MM.png?raw=true" align="center"/></p>
</details>

### Temporal Consistency Module (TCM)
<details>
<summary>Click to expand</summary>

<img src="https://github.com/rafaelpadilla/differentiable-anomaly-detection-pipeline/blob/main/aux_imgs/pipeline_TCM.png?raw=true" align="center"/></p>
</details>

### Classification Module (CM)
<details>
<summary>Click to expand</summary>

<img src="https://github.com/rafaelpadilla/differentiable-anomaly-detection-pipeline/blob/main/aux_imgs/pipeline_CM.png?raw=true" align="center"/></p>
</details>


## Requirements

If you use conda/anaconda, use the file environment.yml to install the needed packages to run the network:
`conda env create -f environment.yml`

## Download dataset

Download the pre-aligned frames using the command:

`ABC.sh`

Optional: You can download the full not aligned videos from the VDAO official web site: [Training set](http://www02.smt.ufrj.br/~tvdigital/database/objects/page_01.html) [Testing set](http://www02.smt.ufrj.br/~tvdigital/database/research/page_01.html). 

## Inform path where the dataset is

Mudar os caminhos do arquivo paths_definitions.py


## Training

Use the command below to train the network:

`python train.py --fold 1 --net DM_MM_TCM_CM`

Use the argument `--fold` to select which fold to train (options: `--fold 1`, `--fold 2`, `--fold 3`, `--fold 4`, `--fold 5`, `--fold 6`, `--fold 7`, `--fold 8`, `--fold 9`).

Use the argument `--net` to train the pipeline with modules placed in different orders (options: `--net  DM_MM_TCM_CM`, `--net DM_TCM_MM_CM`).

Check all possible arguments with the command `python train.py --help`

## Testing

You can download the **pre-trained** models from here:
https://drive.google.com/drive/folders/1-ExzI5REX-Hht-SixjsLJHPZ1S1zpR5y?usp=sharing

## Results

