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
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

categories = ['feminine-coded goods and services',
              'gendered split perception',
              'testimonial injustice',
              'social dominance penalty']

definitions = {
    'women software developers experiences':
        "As a woman software engineer, I’ve faced challenging "
        "situations while collaborating with colleagues in my teams. These include encountering sexism, navigating "
        "a hostile environment, and receiving unequal treatment.",
}

def clean_text(text):
    # Remove special characters and urls
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Perform POS tagging
    tagged_tokens = pos_tag(tokens)

    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token, tag in tagged_tokens:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos))

    # Join cleaned tokens
    cleaned_text = ' '.join(lemmatized_tokens)

    return cleaned_text


############################# Combine csv files #############################
dfs = []
for value in categories:
    df = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/'+value+'.csv')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

############################# Definition Embedding #############################
model = SentenceTransformer('all-mpnet-base-v2')

definitions_embeddings = []
for key, value in definitions.items():
    definitions_embeddings.append(model.encode(value, convert_to_tensor=True))

############################# Data Embedding #############################
df['Cleaned Text'] = df['Text'].apply(clean_text)

# Extract the cleaned text column
text_data = df['Cleaned Text'].tolist()

# Generate sentence embeddings for text data
data_embeddings = model.encode(text_data, convert_to_tensor=True)

############################# Distances to Category #############################
distances = []
for data_embedding in data_embeddings:
    data_embedding = data_embedding.unsqueeze(0)
    data_distances = cosine_similarity(data_embedding.numpy(), definitions_embeddings)
    distances.append(data_distances)

distances = np.array(distances)
closest_definition_index = np.argmax(distances, axis=1)

closest_definitions = []
closest_definition_scores = []

for i, row in enumerate(distances):
    closest_definition_index = np.argmax(row)
    closest_definition_score = row[0][closest_definition_index].item()
    closest_definition = list(definitions.keys())[closest_definition_index]
    closest_definitions.append(closest_definition)
    closest_definition_scores.append(closest_definition_score)

df['Potential Category by SS'] = closest_definitions
df['Potential Category - Cosine Similarity'] = closest_definition_scores

plt.hist(1-df['Potential Category - Cosine Similarity'], bins=20, edgecolor='black')
plt.xlabel('Cosine Distances')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Distances')
plt.show()

df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/Model Scripts/Semantic Similarity/Updated '
          'Definitions Approach 4 - WSE SS Output.csv', index=False)