################################################################################
# Apply Semantic Similarity with Generic Definition of Women Software
# Developers Experiences of Sexism
# By: Amanda Kolopanis
# ! Code is still a work in progress - ** Please use as a guideline ** !
################################################################################

import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, models, util, InputExample, losses, evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score
from sklearn.pipeline import make_pipeline

from sbert_sklearn_wrapper import SklearnSentenceTransformer
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from transformers import AdamW, AutoTokenizer

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

categories = ['feminine-coded goods and services',
              'gendered split perception',
              'testimonial injustice',
              'social dominance penalty']

wsde_definition = {
    'women software developers experiences':
        "As a female software engineer, woman in tech, and woman software developer, I’ve experienced challenging "
        "situations while collaborating with colleagues in my teams. These include encountering sexism and navigating "
        "a hostile environment.",
}

loss_names = ["cosine_similarity"]

def clean_text(text):
    # Remove special characters and urls
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def log_callback_st(train_idx, epoch, training_steps, lr, loss):
    print({
        f"train/{loss_names[train_idx]}_loss": loss,
        f"train/{loss_names[train_idx]}_lr": lr,
        "train/steps": training_steps,
        "train/epoch": epoch
    })


def custom_collate(inner_batch):
    embedding = [item.texts[0] for item in inner_batch]
    labels = [example.label for example in inner_batch]

    # Convert embeddings and labels to tensors
    # embeddings_tensor = torch.tensor(embedding)
    # labels_tensor = torch.tensor(labels)

    return embedding, labels

############################# Combine csv files #############################
# print("Step 1: Combining CSV Data Files")
# dfs = []
# for value in categories:
#     df = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/'+value+'.csv')
#     dfs.append(df)
#
# df = pd.concat(dfs, ignore_index=True)


############################## Labeled Data ####################################
print("Step 2: Extracting Labeled Data")
df_labeled = pd.read_csv(
    'C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/WIP Full Text EDI Classification.csv',
    encoding="iso-8859-1")

X = df_labeled['Text'].apply(clean_text).to_numpy()
y = df_labeled['Taxonomy Label'].to_numpy()

assert len(X) == len(y), "The number of sentences and labels must be equal."

############################# Load Model #############################
print("Step 3: Load Model")
sbert = SklearnSentenceTransformer('taxonomy_model_FULL_TEXT')

######################### Permutation Test Score #########################
print("Step 5: Permutation Test Score")

clf = LogisticRegression(max_iter=1000)  # Ensure enough iterations for convergence

# Custom scorer for F1-score
f1_scorer = make_scorer(f1_score, average='micro')

pipeline = make_pipeline(sbert, clf)

# Perform permutation test score
score, perm_scores, pvalue = permutation_test_score(
    pipeline, X, y, cv=2, scoring=f1_scorer, n_permutations=100, n_jobs=2
)

print(f"Score: {score}")
print(f"Permutation scores: {perm_scores}")
print(f"P-value: {pvalue}")

# Plotting
plt.hist(perm_scores, bins=20, density=True)
plt.axvline(score, ls="--", color="r")

# Adding text
score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
plt.text(0.7 * max(perm_scores), 10, score_label, fontsize=12)

# Adding labels
plt.xlabel("Micro F1-Score score")
plt.ylabel("Probability density")

# Display the plot
plt.show()