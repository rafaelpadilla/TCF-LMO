# Anomaly Detection in Moving-Camera Videos Using Deep-Learning Networks and Fully Differentiable Morphological Operations (DL-DMO)
Anomaly Detection Network is a network made with dedicated modules to process frames and identify the presence of anomalies.

* Overview
* Requirements
* Training
* Testing 
* Results
   * Metrics
   * Frame-level results 
   * Object-level results 
* Acknowledgement
  

## Overview
The anomaly detection network consists of five modules as illustrated below:

<!--- scheme --->
<p align="center">
<img src="https://github.com/rafaelpadilla/dl-smo/blob/main/aux_imgs/pipeline_outputs.png?raw=true" align="center"/></p>

Where: 

(a) aligned reference frame  
(b) target frame (where the anomaly is present)  
(c) output of the dissimilarity module (DM)  
(d) output of the temporal consistency module (TCM)  
(e) eroded version of (d) computed by differentiable morphology module (MM)  
(f) eroded version of (e) computed by differentiable morphology module (MM)  

## Requirements

### Installing packages and libraries:

If you use conda/anaconda, use the file environment.yml to install the needed packages to run the network:
`conda env create -f environment.yml`

## Replicating our results

You can train the model from scratch, or use the pretrained model to evaluate the VDAO testing videos.

Optionally, you could download the full **not aligned** videos from the VDAO official web site: [training set](http://www02.smt.ufrj.br/~tvdigital/database/objects/page_01.html) and [testing set](http://www02.smt.ufrj.br/~tvdigital/database/research/page_01.html).Thus, you could apply alternative alignment/registration techniques and train the DN-LMO model.

### A. (Optional) Train the model from scratch:

Use the commands below to download the datasets (training + validation) and train the network:

**1. Download the aligned dataset for training:** `sh download_training_dataset.sh`    
**2. Download the aligned dataset for validation:** `sh download_testing_dataset.sh`  
**3. Train the network:** `python train.py --fold 1 --net DM_MM_TCM_CM`  
*Notice:* Use the argument `--fold` to select which fold to train (options: `--fold 1`, `--fold 2`, `--fold 3`, `--fold 4`, `--fold 5`, `--fold 6`, `--fold 7`, `--fold 8`, `--fold 9`).

### B. Testing

**1. Download the aligned dataset for testing:** `sh download_testing_dataset.sh`  
**2. Download the pretrained model:** `sh download_pretrained.sh`  
**3. Evaluate the network:** `sh evaluate.sh`

## Results

### Metrics

<details>
<summary>Click to expand</summary>

Our results are compared against previous works in the same database using the true positive rate (TPR), false positve rate (FPR) and DIS, which is the minimum distance of an operating point to the point of ideal behaviour of a ROC curve, as illustrated below

<p align="center">
<img src="https://github.com/rafaelpadilla/dl-smo/blob/main/aux_imgs/ROC_curve.png?raw=true" align="center"/></p>

The best value is obtained when TPR=1 and FPR=0 resulting in a DIS=0, which represents the best possible classification. And it is computed as:

<p align="center">
<img src="https://github.com/rafaelpadilla/dl-smo/blob/main/aux_imgs/eq_DIS.png?raw=true" align="center"/></p>

 </details>

### Frame-level results 

<table>
<thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="3">average</th>
    <th colspan="3">overall</th>
  </tr>
  <tr>
    <th>TPR</th>
    <th>FPR</th>
    <th>DIS</th>
    <th>TPR</th>
    <th>FPR</th>
    <th>DIS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DAOMC</td>
    <td>0.88</td>
    <td>0.39</td>
    <td>0.49</td>
    <td>0.89</td>
    <td>0.42</td>
    <td>0.43</td>
  </tr>
  <tr>
    <td>ADMULT</td>
    <td>0.76</td>
    <td>0.36</td>
    <td>0.59</td>
    <td>0.78</td>
    <td>0.39</td>
    <td>0.44</td>
  </tr>
  <tr>
    <td>MCBS</td>
    <td>1.00</td>
    <td>0.83</td>
    <td>0.83</td>
    <td>1.00</td>
    <td>0.98</td>
    <td>0.98</td>
  </tr>
  <tr>
    <td>mcDTSR</td>
    <td>0.88</td>
    <td>0.25</td>
    <td>0.36</td>
    <td>0.88</td>
    <td>0.28</td>
    <td>0.30</td>
  </tr>
  <tr>
    <td>CNN+RF</td>
    <td>0.74</td>
    <td>0.25</td>
    <td>0.48</td>
    <td>0.75</td>
    <td>0.27</td>
    <td>0.37</td>
  </tr>
  <tr>
    <td><b>DL-DMO (ours)</td>
    <td>0.85</td>
    <td><b>0.18</b></td>
    <td><b>0.33</b></td>
    <td>0.86</td>
    <td><b>0.21</b></td>
    <td><b>0.25</b></td>
  </tr>
</tbody>
</table>

### Object-level results
  
<table>
<thead>
  <tr>
    <th rowspan="2"></th>
    <th colspan="3">average</th>
    <th colspan="3">overall</th>
  </tr>
  <tr>
    <th>TPR</th>
    <th>FPR</th>
    <th>DIS</th>
    <th>TPR</th>
    <th>FPR</th>
    <th>DIS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DAOMC</td>
    <td>0.81</td>
    <td>0.42</td>
    <td>0.53</td>
    <td>0.82</td>
    <td>0.42</td>
    <td>0.45</td>
  </tr>
  <tr>
    <td>ADMULT</td>
    <td>0.70</td>
    <td>0.29</td>
    <td>0.54</td>
    <td>0.72</td>
    <td>0.29</td>
    <td>0.40</td>
  </tr>
  <tr>
    <td>MCBS</td>
    <td>0.88</td>
    <td>0.83</td>
    <td>0.86</td>
    <td>0.89</td>
    <td>0.83</td>
    <td>0.84</td>
  </tr>
  <tr>
    <td>mcDTSR</td>
    <td>0.86</td>
    <td>0.29</td>
    <td>0.39</td>
    <td>0.86</td>
    <td>0.29</td>
    <td>0.32</td>
  </tr>
  <tr>
    <td><b>DL-DMO (ours)</b></td>
    <td>0.85</td>
    <td><b>0.22</b></td>
    <td><b>0.35</b></td>
    <td>0.86</td>
    <td><b>0.22</b></td>
   <td><b>0.26</b></b></td>
  </tr>
</tbody>
</table>

## Acknowledgement

Most of the experimental results reported in this work wereobtained with a Titan X Pascal board gently donated by the NVIDIA Corporation.
