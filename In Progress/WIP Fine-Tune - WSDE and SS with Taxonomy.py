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
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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

taxonomy_definitions = {
    'feminine-coded goods and services':
        "The experiences when women software developers are expected to naturally provide to men because they are "
        "entitled to receive the benefits of women’s goods and services. Moreover, these characteristics are used to "
        "reinforce traditional gender roles. For example, care-mongering is when women are disproportionately "
        "required to be caring and are expected to develop personal relationships with individuals. For example, "
        "I am the only woman in our dev team and I am always implicitly expected to do the administrative tasks "
        "during our meetings. When I confront my team about this, they explain that my organization and note-taking "
        "abilities are a natural talent that benefits the team.",
    'gendered split perception':
        "Women software developers experience harsher judgement when performing the same actions as their male "
        "counterparts even though they have done nothing wrong in moral and social reality. Women may be subject to "
        "moral suspicion and consternation for violating edits of the patriarchal rule book. For example, as a female "
        "software engineer, I feel like my source code is heavily scrutinized by my male teammates. When I submit "
        "similar work as my male co-workers, I tend to receive more critiques compared to my colleagues despite our "
        "work being identical in logic and performance.",
    'testimonial injustice':
        "Arises due to systematic biases that afflict women software developers as a social group that has "
        "historically been and to some extent remains unjustly socially subordinate. The group members experiences "
        "challenges as being regarded as less credible when making claims about certain matters, or against certain "
        "people, hence being denied the epistemic status of knowers. For example, I am a woman software developer. I "
        "find that when I present an ideas to my development team, they often ignore my input. However, when my male "
        "colleague repeats the same ideas in a follow-up meeting, the team almost immediately accepts them.",
    'social dominance penalty':
        "People are (often unwittingly) motivated to maintain gender hierarchies by applying social penalties to "
        "women software developers who compete for, or otherwise threaten to advance to, high-status, masculine-coded "
        "positions. This is experienced when women in such positions who are agentic are perceived as extreme in "
        "masculine-coded traits like being arrogant and aggressive. For example, as one of the female programmers in "
        "our team, I sometimes experience a sense of hostility when I provide constructive criticism or potential "
        "improvements to my male counterparts' source code. I give the same type of feedback to my female colleagues "
        "and receive praises."
}

loss_names = ["cosine_similarity"]


def clean_text(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()

    tokens = word_tokenize(text.lower())
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
print("Step 1: Combining CSV Data Files")
dfs = []
for value in categories:
    df = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/'+value+'.csv')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


############################## Labeled Data ####################################
print("Step 2a: Extracting Labeled Data")
df_labeled = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/WIP EDI Examples.csv')


############################# Definition Embedding #############################
print("Step 2b: Define Model")
word_embedding_model = SentenceTransformer('all-mpnet-base-v2')
pooling_model = models.Pooling(word_embedding_model.get_sentence_embedding_dimension(), pooling_mode="max")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


############################ Fine-Tune Model ####################################
print("Step 3: Fine-Tuning Model on WSDE")
tokenizer = word_embedding_model.tokenizer

encoded_texts = tokenizer(clean_text(df_labeled['Text']))
labels = torch.tensor([float(label) for label in df_labeled['WSDE Label'].tolist()])

train_examples = []
sentences1 = []
sentences2 = []
scores = []
for _, row in df_labeled.iterrows():
    train_examples.append(InputExample(texts=[clean_text(row['Text']),
                                              clean_text(wsde_definition['women software developers experiences'])],
                                       label=float(row['WSDE Label'])))
    sentences1.append(clean_text(row['Text']))
    sentences2.append(clean_text(wsde_definition['women software developers experiences']))
    scores.append(float(row['WSDE Label']))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=25)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=3,
    evaluator=evaluator,
    evaluation_steps=5
)


############################# Definition Embedding #############################
print("Step 4: WSDE Definition Embedding")
definitions_embeddings = []
for key, value in wsde_definition.items():
    definitions_embeddings.append(model.encode(value, convert_to_tensor=True))


############################# Data Embedding #############################
print("Step 5: Data Embedding")

# Removing data points from EDI Examples.csv (df_labeled) to avoid data leakage
df = df.drop_duplicates(subset=['SubReddit', 'ID'])
indices_to_drop = df[df.apply(lambda row: (row['SubReddit'], row['ID']) in zip(df_labeled['SubReddit'], df_labeled['ID']), axis=1)].index
df.drop(indices_to_drop, inplace=True)

df['Cleaned Text'] = df['Text'].apply(clean_text)

# Extract the cleaned text column
text_data = df['Cleaned Text'].tolist()

# Generate sentence embeddings for text data
data_embeddings = model.encode(text_data, convert_to_tensor=True)

############################# Distances to Category #############################
print("Step 6: Calculate Data Cosine Distance to the Category")
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
    closest_definition = list(wsde_definition.keys())[closest_definition_index]
    closest_definitions.append(closest_definition)
    closest_definition_scores.append(closest_definition_score)


df['Potential Category using Static Keyword Extraction'] = closest_definitions
df['WSDE Cosine Similarity'] = closest_definition_scores

plt.hist(1-df['Potential Category - Cosine Similarity'], bins=20, edgecolor='black')
plt.xlabel('Cosine Distances')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Distances')
plt.show()

############################# Select Data Subset #############################
threshold = 0.4
print("Step 7: Select subset of data where cosine distance <= " + str(threshold))
threshold_df = df[1 - df['WSDE Cosine Similarity'] <= threshold]

df_labeled_1 = df_labeled[df_labeled['WSDE Label'] == 1]
df_labeled_1 = df_labeled_1[
    'SubReddit', 'Hot/Top/New', 'Scraped Date', 'Posted Date', 'Posted by', 'Post or Comment', 'Post Title',
    'ID', 'Text', 'Text Length', 'Potential Category using Static Keyword Extraction',
    'Cleaned Text', 'WSDE Cosine Similarity', 'WSDE Label'
]
threshold_df = pd.concat([threshold_df, df_labeled_1], axis=0)

############################# Definitions Embedding #############################
print("Step 8: Taxonomy Definition Embeddings")
taxonomy_definitions_embeddings = []
for key, value in taxonomy_definitions.items():
    taxonomy_definitions_embeddings.append(model.encode(value, convert_to_tensor=True))

############################# Data Embedding using Paragraphs #############################
print("Step 9: Data Embedding using Paragraphs")
threshold_df['Text per Paragraph'] = threshold_df['Text'].apply(lambda text: text.split('\n\n'))

new_rows = []
for index, row in threshold_df.iterrows():
    paragraphs = row['Text per Paragraph']
    for paragraph_index, paragraph in enumerate(paragraphs):
        new_row = {
            'SubReddit': row['SubReddit'],
            'Hot/Top/New': row['Hot/Top/New'],
            'Scraped Date': row['Scraped Date'],
            'Posted Date': row['Posted Date'],
            'Posted by': row['Posted by'],
            'Post or Comment': row['Post or Comment'],
            'Post Title': row['Post Title'],
            'ID': row['ID'],
            'Text': row['Text'],
            'Text Length': row['Text Length'],
            'Potential Category using Static Keyword Extraction': row['Potential Category using Static Keyword Extraction'],
            'Paragraph': paragraph.strip(),
            'Cleaned Text': clean_text(paragraph.strip()),
            'WSDE': row['WSDE'],
            'WSDE Cosine Similarity': row['WSDE Cosine Similarity'],
            'Label': row['WSDE Label']
        }
        new_rows.append(new_row)
        print(f"Processed paragraph {paragraph_index + 1} for row {index}")

# Create the new DataFrame
threshold_paragraph_df = pd.DataFrame(new_rows)
taxonomy_text_data = threshold_paragraph_df['Cleaned Text'].tolist()
taxonomy_data_embeddings = model.encode(taxonomy_text_data, convert_to_tensor=True)

############################# Distances per Category #############################
print("Step 10: Calculate Data Cosine Distance to the Category")
taxonomy_distances = []
for data_embedding in taxonomy_data_embeddings:
    data_embedding = data_embedding.unsqueeze(0)
    data_distances = cosine_similarity(data_embedding.numpy(), taxonomy_definitions_embeddings)
    taxonomy_distances.append(data_distances)

taxonomy_distances = np.array(taxonomy_distances)
taxonomy_closest_definition_index = np.argmax(taxonomy_distances, axis=1)

taxonomy_closest_definitions = []
taxonomy_closest_definition_scores = []

# Find the closest definition for each sample
for i, row in enumerate(taxonomy_distances):
    taxonomy_closest_definition_index = np.argmax(row)
    closest_definition_score = row[0][taxonomy_closest_definition_index].item()
    closest_definition = list(taxonomy_definitions.keys())[taxonomy_closest_definition_index]
    taxonomy_closest_definitions.append(closest_definition)
    taxonomy_closest_definition_scores.append(closest_definition_score)

threshold_paragraph_df['Taxonomy Category by SS'] = taxonomy_closest_definitions
threshold_paragraph_df['Taxonomy Category - Cosine Similarity'] = taxonomy_closest_definition_scores

threshold_paragraph_df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/Taxonomy '
          'Combined - WSDE and Taxonomy Definitions.csv', index=False)