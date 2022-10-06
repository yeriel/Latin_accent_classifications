# Latin accent classifications

Audio classification is one of the most widely used applications in Deep Learning. It involves learning to classify sounds and predict the category of that sound. This type of problem can be applied to many practical scenarios such as the recognition of accents of different nationalities and whether it belongs to a woman or a man.

## Getting Started

Copy the repository and create a folder with the name dataset.

```
git clone https://github.com/yeriel/Latin_accent_classifications.git
cd Latin_accent_classifications
mkdir dataset
```
Then download the zip file with the dataset from kaggle at the following [link](https://www.kaggle.com/competitions/clasificacion-de-acentos-latinos/data)

The downloaded zip file move it into the dataset folder just created then using python run the preprocessing script which is part of the repository. Once executed it is possible to train some of the models from the models file.

```
python preprocessing.py
```