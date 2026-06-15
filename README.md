\# Malaria Species Classification using Deep Learning



\## Overview



This repository contains implementations of deep learning models for automated malaria species classification from thin blood smear microscopy images.



The repository includes multiple CNN and Transformer-based architectures for comparative evaluation.



\## Implemented Models



\- HCAST-3Level-ResNet

\- HCAST-Swin

\- HCAST-ViT

\- Swin Transformer

\- Vision Transformer (ViT)

\- InceptionV3



\## Dataset



The experiments were conducted on thin blood smear microscopy images containing:



\- Plasmodium vivax

\- Plasmodium falciparum



\## Requirements



\- Python 3.10+

\- TensorFlow

\- NumPy

\- Pandas

\- Scikit-learn

\- Matplotlib

\- OpenCV



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Training



Example:



```bash

python HCAST\_ViT.py

```



or



```bash

python HCAST\_Swin.py

```



\## Evaluation Metrics



The following metrics were used:



\- Accuracy

\- Sensitivity

\- Specificity

\- Precision

\- F1-Score

\- MCC

\- G-Mean

\- AUC



\## Repository Structure



```text

README.md

HCAST\_3level\_RESNET.py

HCAST\_Swin.py

HCAST\_ViT.py

SwinTransformer.py

ViT.py

inceptionv3new.py

```



\## Authors



Ghufran Alam Siddiqui



\## License



For research and academic purposes only.

