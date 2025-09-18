# Project - CNN on ViT

This folder contains the code to run the CNN on ViT paper implementation. 

## Description

CNN on ViT implements a mechanism called Explicitly Class Specific Boundaries for domain adaptation/

## Getting Started

### Dependencies

* Python 3

### Dataset Download

* There are 3 datasets used to test the code - OfficeHome, Office31, Domainnet
* OfficeHome dataset can be downloaded from the following link: https://www.hemanthdv.org/officeHomeDataset.html
* Command to download the Office31 Dataset: 
```
pip install -U --no-cache-dir gdown --pre
gdown --id 0B4IapRTv9pJ1WGZVd1VDMmhwdlE
tar -xvf domain_adaptation_images.tar.gz
```
* Domainnet can be downloaded from the following link: https://ai.bu.edu/M3SDA/
* The code to download domainnet and office31 is available in the attached cnn-on-vit-sample ipynb
### Executing program

* Download the dataset required.
* Use the commands given in the cnn-on-vit-sample ipynb and setup the Conda environment
```
conda env create -f environment.yml
```
* Import the dataset to the kaggle working directory
* Update the train.yaml file with the required configurations ( Source and Target domain, Number of epochs, Batch size etc) and replace it in the kaggle working directory with the required one
* Run the command to train the model given in the sample ipynb file
```
!source activate ecb && echo "Activated: $CONDA_DEFAULT_ENV" && python train.py --cfg configs/train.yaml
```
