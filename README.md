# Fake_News_Detection_Project
a simple GUI application that uses a neural network to classify news articles as real or fake


## Running this program!

1. clone this repository and open as a project in Pycharm 
2. Install all libaries listed in requirements.txt
3. set main.py as your run configuration.
4. run main.py
5. enter any news article title into the title box and the main text into the text box (use ctrl+v to paste)
6. click analyze.
7. view the data analysis graphs along with the model's prediction
8. click the 'close' button or 'try another' button to analyze a second article.


## Model Details

### Model Type
  -- The model used is built on a pretrained BERT model that's been retasked for news classification using PyTorch combined with Transformers by HuggingFace.
  
### Training Data
  -- The training data used is from this url: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
