# This README serves as design document 

## Project documentation

The goal of this poject is to classify amazon.com reviews between negative and positive. See "Dataset description" section for dataset overview. It utilises fine-tuned BERT model to classify reviews between positive and negative. BERT was trained on Google Colab, see file ```LSML2_Abilov_Alikhan.ipynb```. After beeing trained, pickle file was extracted. To download pickle file, see "How to reproduce" section. After that, HTML frontend was developed using FastAPI. This API with pickle file and other files is then composed into docker file (see ```Dockerfile```). The overview of project is as follow:

1) Design document and dataset description - README file
2) Model training code - Jupyter Notebook
3) docker file - synchronous project
4) client - HTML Frontend
5) model - BERT transfer learning

Below is description of files:

1)
 
## How to reproduce 

1) Dowload everything as one directory
2) Download pickle file that was produces by the BERT (this pickle file is too big for github repo, you can dowload it from here ```https://drive.google.com/file/d/1K32r1BzgzdfVhLOdE-sX-2eNjD-j0h6e/view?usp=sharing```) and place it inside downloaded directory
3) Run command ``` sudo docker build -t abilov:latest -f Dockerfile .``` inside dwnloaded directory
4) When docker image finished building, run ```sudo docker run -p 80:80 abilov:latest```
5) Follow this link in your browser ```http://0.0.0.0:80```, you will see HTML frontend with field to write in
6) Write review, wait till it finished and see answer (either "Negative review" or "Positive review")

## Network description



## Dataset description

Datasets were taken from here https://www.kaggle.com/datasets/yacharki/amazon-reviews-for-sa-binary-negative-positive-csv?select=amazon_review_sa_binary_csv

It's Amazon reviews dataset for sentiment analysis (1.74G). It's already divided into train and test datasets. Each dataset has 3 columns. Class index (1 or 2) which represents if it's positive or negaive sentiment, review title and review text. Samples are comma-sparated. Each class has 1,800,000 training samples and 200,000 testing samples. Unfortunatly, this dataset was too big for Google Colav to handle, so only part of it was used.
