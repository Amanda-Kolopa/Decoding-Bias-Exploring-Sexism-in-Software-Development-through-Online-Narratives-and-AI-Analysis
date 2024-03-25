################################################################################
# Apply Semantic Similarity with Taxonomy Definitions
# By: Amanda Kolopanis
# ! Code is still a work in progress - ** Please use as a guideline ** !
################################################################################

import re
import nltk
import pandas as pd
import numpy as np
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

definitions = {
    'feminine-coded goods and services':
        "The experiences when women software developers are expected to naturally provide to men because they are "
        "entitled to receive the benefits of women’s goods and services. Moreover, these characteristics are used to "
        "reinforce traditional gender roles. For example, care-mongering is when women are disproportionately "
        "required to be caring and are expected to develop personal relationships with individuals.",
    'gendered split perception':
        "Women software developers experience harsher judgement when performing the same actions as their male "
        "counterparts even though they have done nothing wrong in moral and social reality. Women may be subject to "
        "moral suspicion and consternation for violating edits of the patriarchal rule book.",
    'testimonial injustice':
        "Arises due to systematic biases that afflict women software developers as a social group that has "
        "historically been and to some extent remains unjustly socially subordinate. The group members experiences "
        "challenges as being regarded as less credible when making claims about certain matters, or against certain "
        "people, hence being denied the epistemic status of knowers.",
    'social dominance penalty':
        "People are (often unwittingly) motivated to maintain gender hierarchies by applying social penalties to "
        "women software developers who compete for, or otherwise threaten to advance to, high-status, masculine-coded "
        "positions. This is experienced when women in such positions who are agentic are perceived as extreme in "
        "masculine-coded traits like being arrogant and aggressive."
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
# TODO: set threshold to desired value
threshold = 0.75

df = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/Model Scripts/Semantic Similarity/Updated '
          'Definitions Approach 4 - WSDE.csv')

threshold_df = df[1 - df['WSDE Cosine Similarity'] <= threshold]

############################# Definitions Embedding #############################
model = SentenceTransformer('all-mpnet-base-v2')

definitions_embeddings = []
for key, value in definitions.items():
    definitions_embeddings.append(model.encode(value, convert_to_tensor=True))

############################# Data Embedding #############################
threshold_df['Cleaned Text'] = threshold_df['Text'].apply(clean_text)

text_data = threshold_df['Cleaned Text'].tolist()
data_embeddings = model.encode(text_data, convert_to_tensor=True)

############################# Distances per Category #############################
distances = []
for data_embedding in data_embeddings:
    data_embedding = data_embedding.unsqueeze(0)
    data_distances = cosine_similarity(data_embedding.numpy(), definitions_embeddings)
    distances.append(data_distances)

distances = np.array(distances)
closest_definition_index = np.argmax(distances, axis=1)

closest_definitions = []
closest_definition_scores = []

# Find the closest definition for each sample
for i, row in enumerate(distances):
    closest_definition_index = np.argmax(row)
    closest_definition_score = row[0][closest_definition_index].item()
    closest_definition = list(definitions.keys())[closest_definition_index]
    closest_definitions.append(closest_definition)
    closest_definition_scores.append(closest_definition_score)

threshold_df['Potential Category by SS'] = closest_definitions
threshold_df['Potential Category - Cosine Similarity'] = closest_definition_scores

threshold_df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/Model Scripts/Semantic Similarity/Taxonomy '
          'Definitions - Second Phase.csv', index=False)