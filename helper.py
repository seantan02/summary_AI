from collections import Counter
import re
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import unicodedata
import string 
import json
import torch
from pathlib import Path
from shared import logger
import nltk
import inflect

# Check if the wordnet corpus is already downloaded
try:
    nltk.data.find('corpora/wordnet.zip')
    print("Wordnet is already downloaded.")
    nltk.data.find('corpora/punkt.zip')
    print("Punkt is already downloaded.")
except LookupError:
    print("Wordnet not found. Downloading now...")
    nltk.download('wordnet')
    print("Punkt not found. Downloading now...")
    nltk.download('punkt')

from nltk.stem.wordnet import WordNetLemmatizer

ALL_LETTERS = string.ascii_letters + " .,;'-"

"""
All helper functions mainly for Seq2Seq model and text processing begins here:
"""


def plot_loss(training_loss: List, evaluation_loss: List, epoch_range: List, output_path: Path):
    # Create the plot
    assert len(training_loss) == len(evaluation_loss) == len(epoch_range), f"Length of training_loss, evaluation_loss and epoch_range should be the same. Got {len(training_loss)}, {len(evaluation_loss)} and {len(epoch_range)} respectively."
    plt.figure(figsize=(10, 6))

    plt.plot(epoch_range, training_loss, label='Training Loss', marker='o')
    plt.plot(epoch_range, evaluation_loss, label='Evaluation Loss', marker='o')

    # Add titles and labels
    plt.title('Training and Evaluation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend()

    # Create the parent directory if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plt.savefig(output_path, format='jpg')


def save_vocabulary(w2t, t2w, vocab_dir_path: Path, type: str):
    vocab_dir_path.mkdir(parents=True, exist_ok=True)

    w2t_path = vocab_dir_path / 'output_w2t.json'
    t2w_path = vocab_dir_path / 'output_t2w.json'
    
    if type == "input":
        w2t_path = vocab_dir_path / 'input_w2t.json'
        t2w_path = vocab_dir_path / 'input_t2w.json'
    
    with open(w2t_path, 'w') as f:
        json.dump(w2t, f)
    
    with open(t2w_path, 'w') as f:
        json.dump(t2w, f)

    logger.info(f"Vocabulary version saved at {vocab_dir_path}")


def load_vocabulary(vocab_dir_path: Path, type:str):
    
    if not vocab_dir_path.exists():
        return None, None
    
    w2t_path = vocab_dir_path / 'output_w2t.json'
    t2w_path = vocab_dir_path / 'output_t2w.json'
    
    if type == "input":
        w2t_path = vocab_dir_path / 'input_w2t.json'
        t2w_path = vocab_dir_path / 'input_t2w.json'

    if not t2w_path.exists() or not w2t_path.exists():
        return None, None

    with w2t_path.open(mode="r") as f:
        w2t = json.load(f)
    
    with t2w_path.open(mode='r') as f:
        t2w = json.load(f)

    t2w = {int(k): v for k, v in t2w.items()}  # Convert keys to int

    logger.info(f"Vocabulary loaded from {vocab_dir_path}")

    return w2t, t2w


def save_model(model, 
               checkpoint_path: Path):

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True) # Create the parent directory if it does not exist
    
    checkpoint = {
        'model_kwargs': model.kwargs,
        'model_state_dict': model.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Model saved at {checkpoint_path}")


def save_optimizer(optimizer, 
               loss, 
               epoch, 
               checkpoint_path: Path):

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True) # Create the parent directory if it does not exist

    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Optimizer saved at {checkpoint_path}")

def load_model(model_class, 
               checkpoint_path: Path,
               device):
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = model_class(**checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model version loaded from {checkpoint_path}")

        return model
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}")
        return None


def load_optimizer(optimizer, 
               checkpoint_path: Path,
               device):
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        loss = checkpoint['loss']

        logger.info(f"Model optimizer version loaded from {checkpoint_path}")

        return optimizer, start_epoch, loss
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}")
        return optimizer, 0, None


def tokenize_words(text: str, 
                   w2t, 
                   lower_case=False, 
                   convert_unicode_to_ascii=False, 
                   stop_words={},
                   lemmatize_words=False, 
                   singularize_words=False) -> torch.Tensor:
    """
    This function tokenize the text by lowering the text case, remove special characters and remove stop words, and word by word
    """
    processed_text = preprocess_text(text=text, 
                                lower_case=lower_case, 
                                convert_unicode_to_ascii=convert_unicode_to_ascii, 
                                stop_words=stop_words,
                                lemmatize_words=lemmatize_words,
                                singularize_words=singularize_words)
    
    text = "<SOS> "+processed_text+" <EOS>"

    result = []

    for word in text.split(" "):
        word = word.replace(" ", "")

        if word not in w2t:
            result.append(w2t["<UNK>"])
            continue

        result.append(w2t[word])

    return torch.tensor(result)


def update_vocabulary(old_w2t, old_t2w, new_texts, lower_case=False, convert_unicode_to_ascii=False, stop_words={}, lemmatize_words=False, singularize_words=False):

    # Start with the existing vocabulary
    new_w2t = old_w2t.copy()
    new_t2w = old_t2w.copy()

    text = ""
    
    for sentence in new_texts:
        text += preprocess_text(text=sentence, 
                                lower_case=lower_case, 
                                convert_unicode_to_ascii=convert_unicode_to_ascii, 
                                stop_words=stop_words,
                                lemmatize_words=lemmatize_words,
                                singularize_words=singularize_words)+" "
    
    max_token_id = len(old_w2t)  # Starting token id

    for word in text.split(" "):
        word = word.strip()
        word = word.replace(" ", "")

        if len(word) <= 1:
            continue

        if word not in new_w2t:
            new_w2t[word] = max_token_id
            new_t2w[max_token_id] = word
            max_token_id += 1
    
    return new_w2t, new_t2w


def get_vocabulary(document: List[str], lower_case=False, convert_unicode_to_ascii=False, stop_words={}, lemmatize_words=False, singularize_words=False, minimum_occurance=1):
    """
    This function tokenize the text by NOT lowering the text case, remove special characters and remove stop words, and word by word
    """

    text = ""

    for sentence in document:
        text += preprocess_text(text=sentence, 
                                lower_case=lower_case, 
                                convert_unicode_to_ascii=convert_unicode_to_ascii, 
                                stop_words=stop_words,
                                lemmatize_words=lemmatize_words,
                                singularize_words=singularize_words)+" "

    vocab = set()

    add_later = ["<PAD>", "<SOS>", "<EOS>"]

    all_words = text.split(" ")
    word_counts = Counter(all_words)

    for word in all_words:
        if word in add_later:
            continue

        if word_counts[word] < minimum_occurance:
            continue

        word = word.strip()
        word = word.replace(" ", "")

        if len(word) == 0:
            continue
        
        vocab.add(word)

    w2t = dict()
    t2w = dict()

    w2t["<PAD>"] = 0
    w2t["<SOS>"] = 1
    w2t["<EOS>"] = 2
    w2t["<UNK>"] = 3
    t2w[0] = "<PAD>"
    t2w[1] = "<SOS>"
    t2w[2] = "<EOS>"
    t2w[3] = "<UNK>"
    
    starting_token_id = len(t2w)

    for i, word in enumerate(vocab, starting_token_id):
        w2t[word] = i
        t2w[i] = word

    return w2t, t2w

def preprocess_text(text: str, lower_case=False, convert_unicode_to_ascii=False, stop_words = None, lemmatize_words=False, singularize_words=False)->str:

    if lower_case:
        text = text.lower()

    if convert_unicode_to_ascii:
        text = unicode_to_ascii(text)

    cleaned_text = re.sub(r"'t", 't', text)
    cleaned_text = re.sub(r"'re", ' are', cleaned_text)
    cleaned_text = re.sub(r"'s", ' is', cleaned_text)
    cleaned_text = re.sub(r"'d", ' would', cleaned_text)
    cleaned_text = re.sub(r"'ll", ' will', cleaned_text)
    cleaned_text = re.sub(r"'ve", ' have', cleaned_text)
    cleaned_text = re.sub(r"'m", ' am', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s()]|[\n\r]', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace("cannot", "can not")  # Allow can and not to be separated
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    final_text = []

    inflect_engine = inflect.engine()

    for clean_text in cleaned_text.split(" "):
        clean_text = clean_text.strip()

        if len(clean_text) == 0:
            continue

        if len(clean_text) == 1 and clean_text not in ['a', 'i']:  # Only a is a valid single character word
            continue

        if lemmatize_words:
            clean_text = WordNetLemmatizer().lemmatize(clean_text, "v")

        final_word = None
        if singularize_words:
            final_word = inflect_engine.singular_noun(clean_text)

        final_word = final_word if final_word else clean_text
        
        if stop_words is not None and final_word in stop_words:  # Remove stop words
            continue

        final_text.append(final_word)
    
    return " ".join(final_text)
    

def unicode_to_ascii(s: str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

"""
All helper functions mainly for Seq2Seq model and text processing ends here
"""

def most_common_words(text, n=10):
    # Use regex to find words, convert to lower case
    words = re.findall(r'\b\w+\b', text.lower())
    # Count occurrences of each word
    word_counts = Counter(words)
    # Get the n most common words
    return word_counts.most_common(n)

def first_part_clean_error_info(sentences: pd.Series):
  
    result = []

    for sentence in sentences:
        sentence = re.split(r'\r', sentence, 1)[0]
        if sentence == "":
            raise Exception("ERROR: Sentence after splitting is empty")
        
        result.append(preprocess_text(sentence))
    
    return result

def last_clean_error_info(sentences: pd.Series):
  
    result = []

    for sentence in sentences:
        sentence = re.split(r'\n', sentence, 1)[-1]
        if sentence == "":
            raise Exception("ERROR: Sentence after splitting is empty")
        
        result.append(preprocess_text(sentence))
    
    return result

def all_clean_error_info(sentences: pd.Series):
  
    result = []

    for sentence in sentences:
        if sentence == "":
            raise Exception("ERROR: Sentence after splitting is empty")
        
        result.append(preprocess_text(sentence))
    
    return result

def get_embeddings_from_row_non_zero_avg(vectorized_documents):
    embeddings = []

    for i in range(vectorized_documents.shape[0]):
        total = 0
        for index in vectorized_documents.getrow(i).nonzero()[1]:
            total += vectorized_documents.getrow(i)[0, index]
        
        embeddings.append([total / len(vectorized_documents.getrow(i).nonzero()[1])])

    return np.asarray(embeddings)  # Average the word embeddings for each document

def produce_wcss_diagram_in_range(num_clusters_range, vectorized_documents, image_output_path="'wcss_elbow_plot.png'"):
    # Below code is to find the optimal number of clusters using the Elbow method
    wcss = []

    for i in num_clusters_range:
        kmeans = kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=100, random_state=2002)

        kmeans.fit(vectorized_documents)

        # Compute Within-Cluster Sum of Squares (WCSS)
        wcss.append(kmeans.inertia_)
    # Plot the WCSS against the number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(num_clusters_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.xticks(num_clusters_range)
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(image_output_path)

def get_top_terms(cluster_centers, terms, top_n=10):
        top_terms = {}
        for i, center in enumerate(cluster_centers):
            term_indices = center.argsort()[-top_n:][::-1]
            top_terms[i] = [(terms[ind], center[ind]) for ind in term_indices]
        return top_terms