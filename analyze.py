import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
import contractions
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter


STOP_WORDS_PATH = 'extra_files/English_Stop_Words_List.txt'
PRETRAINED = 'Model/BertPreTrained'
PUNC = re.compile(r"[^\w\s]+")
URLS = re.compile(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+"
                  r"[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+["
                  r"a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-"
                  r"zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# gather the stop words into a list
with open(STOP_WORDS_PATH, 'r') as f:
    stop_words = f.read()
    stop_words = stop_words.split(",")
    stop_words = [i.replace('"', "").strip() for i in stop_words]

def clean_text(text):
    """
    Expands contractions, removes URLs and punctuation symbols and removes all stop words from text.
    :param text: string
    :return: cleaned_text string
    """

    cleaned_text = [contractions.fix(word) for word in text.split()]
    cleaned_text = ' '.join(cleaned_text)
    cleaned_text = URLS.sub('', str(cleaned_text))
    cleaned_text = ' '.join([item for item in cleaned_text.split() if item not in stop_words])
    cleaned_text = PUNC.sub('', str(cleaned_text))

    return cleaned_text


def load_model():
    """
    Loads pretrained pytorch model and maps the device to CPU
    :return: BERT model torch object
    """
    device = torch.device('cpu')

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False)
    model.to(device)

    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # model.save_pretrained('Model/BertPreTrained')
    return model


def build_wordcloud(text, width, height):
    """
    Creates and returns a word cloud using input text and size specifications
    """
    text_list = text.split()
    counts = Counter(text_list)
    wordcloud = WordCloud(background_color='black',
                          width=width, height=height,
                          max_words=15, relative_scaling=1)
    plt.title('Important Words', fontsize=48)

    wordcloud_image = wordcloud.fit_words(counts)

    return wordcloud_image


def build_tf_graph(text):
    """
    Builds a term frequency data used to make a bar chart
    :param text: cleaned_text string
    :return: a 2 dimensional list: [[top 3 words], [frequency of the top 3 words]]
    """

    text_list = text.split()
    counts = Counter(text_list)

    labels, values = zip(*counts.items())

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    top_3_labels = labels[0:3]
    top_3_values = values[0:3]

    return [top_3_labels, top_3_values]


def build_probability_graph(probability):
    """
    calculates the confidence probability of a model's prediction
    :param probability: the output of the models predictions run through a softmax layer
    :return: a tuple (fake probability, true probability)
    """
    zero_prob = probability[0].detach().numpy()
    one_prob = zero_prob[1]
    zero_prob = zero_prob[0]

    label_size = (zero_prob, one_prob)

    return label_size


def tokenize_map(article):
    """
    creates tokens to feed into the model for analysis
    :param article: the entire cleaned text of a news article. String
    :return: tokens for the model to read.
    """

    input_ids = []
    attention_masks = []
    for text in article:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # [CLS] & [SEP]
            truncation='longest_first',  # Control truncation
            max_length=100,  # Max length about texts
            padding='max_length',  # Pad and truncate about sentences
            return_attention_mask=True,  # Attention masks
            return_tensors='pt')  # Return to pytorch tensors
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def analyze(title, input_text):
    """
    builds all graph data and model predictions
    :param title: Article title, string
    :param input_text: Article text, string
    :return: cleaned version of input text, Term Frequency data, a tuple of probabilities, and the model's prediction
    """
    model = load_model()
    device = torch.device('cpu')

    text = title + ' ' + input_text

    cleaned_text = clean_text(text)

    tf_graph = build_tf_graph(cleaned_text)

    input_id, attention_mask = tokenize_map([cleaned_text])
    attention_mask = attention_mask.unsqueeze(0)

    g_label = model(input_id.to(device), token_type_ids=None, attention_mask=attention_mask.to(device)[0])

    pred = torch.max(g_label.logits, 1)[1][0].item()
    probability = torch.nn.functional.softmax(g_label.logits, dim=1)

    pie_chart = build_probability_graph(probability)

    return cleaned_text, tf_graph, pie_chart, pred
