<!-- TITLE -->
<br />
<p align="center">
  <h3 align="center">Real-time detection for renal pathology</h3>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Presentation of the Project](#presentation-of-the-project)
* [Prerequisite](#prerequisite)
* [Workflow](#workflow)
  * [Dataset](#dataset)
  * [Train](#train)
  * [Test](#test)
  * [Perform detections on video](#perform-detections-on-video)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)


<!-- PRESENTATION OF THE PROJECT -->
## Presentation Of The Project

This research is carried out as part of the project Sys-MIFTA and has been been published in the conference [IEEE 33rd International Symposium on
Computer Based Medical Systems (CBMS)](http://cbms2020.org/). 
The paper is available [here](https://ieeexplore.ieee.org/abstract/document/9183014).
This work is a deep learning project applied to medical images. 
The dataset contains WSI of stained H&E renal nephrectomies that are used by the YOLOv3 network to create a model capable of detecting glomeruli in real-time.


<!-- GETTING STARTED -->
## Prerequisite

* Before executing the sctipts, make sure you have correctly entered the configuration file: **config.cfg**
* The medical images used for this project are private data that we cannot share.
You will need to use your own data. 
Here is the hierarchy that is expected by the workflow to work properly:
![Hierarchy](https://github.com/RobinHCK/keras-yolo3/blob/master/img/ARGD_dataset_hierarchy.png)

<!-- WORKFLOW -->
## Workflow

### 1. Dataset

Perform data augmentation:
* python FromARGDDatasetToAugmentedARGDDataset.py --config config.cfg


### 2. Train

Download the model pretrained on COCO dataset:
* wget https://pjreddie.com/media/files/yolov3.weights
<br/>
Convert the weights for Keras:
* python convert.py yolo3/yolov3.cfg yolov3.weights model_data/yolo.h5
<br/>
Train the network:
* python train.py 
<br/>
*See the configuration file to know the model location*


### 3. Test

Create a file fill with detections performed by the network:
* python test.py
Compute and write metrics in datas.xlsx:
* python ComputeMetricsOnTest.py --config config.cfg
Draw detections on WSI with the best F1Score per scale:
* python DrawBestWSI.py --config config.cfg
![Neph](https://github.com/RobinHCK/keras-yolo3/blob/master/img/nephrectomy_with_detections.png)
Draw graphics thanks to datas.xlsx:
* python DrawGraphics.py
*Do not forget to test the right model, see the configuration file to know the model location*


### 4. Perform Detections On Video

* python yolo_video.py --input video/your_video.mp4 --output video/your_video_with_detections.mp4 --model model_data/yolo.h5
![Video](https://github.com/RobinHCK/keras-yolo3/blob/master/img/biopsy_with_detections.jpg)
*Do not forget to test the right model, see the configuration file to know the model location*


<!-- CONTACT -->
## Contact

Robin Heckenauer - robin.heckenauer@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* ERACoSysMed project "SysMIFTA", co-funded by EU H2020 and the national funding agencies German Ministry of Education and Research (BMBF) project management PTJ (FKZ: 031L-0085A), and Agence National de la Recherche (ANR), project number ANR-15-CMED-0004.
* The High Performance Computing center of the University of Strasbourg. The computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d'Avenir) and the CPER Alsacalcul/Big Data.
