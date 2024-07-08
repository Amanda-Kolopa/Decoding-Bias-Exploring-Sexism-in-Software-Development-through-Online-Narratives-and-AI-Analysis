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

# taxonomy_definitions = {
#     'feminine-coded goods and services':
#         "The experiences when women software developers are expected to naturally provide to men because they are "
#         "entitled to receive the benefits of women’s goods and services. Moreover, these characteristics are used to "
#         "reinforce traditional gender roles. For example, care-mongering is when women are disproportionately "
#         "required to be caring and are expected to develop personal relationships with individuals.",
#     'gendered split perception':
#         "Women software developers experience harsher judgement when performing the same actions as their male "
#         "counterparts even though they have done nothing wrong in moral and social reality. Women may be subject to "
#         "moral suspicion and consternation for violating edits of the patriarchal rule book.",
#     'testimonial injustice':
#         "Arises due to systematic biases that afflict women software developers as a social group that has "
#         "historically been and to some extent remains unjustly socially subordinate. The group members experiences "
#         "challenges as being regarded as less credible when making claims about certain matters, or against certain "
#         "people, hence being denied the epistemic status of knowers.",
#     'social dominance penalty':
#         "People are (often unwittingly) motivated to maintain gender hierarchies by applying social penalties to "
#         "women software developers who compete for, or otherwise threaten to advance to, high-status, masculine-coded "
#         "positions. This is experienced when women in such positions who are agentic are perceived as extreme in "
#         "masculine-coded traits like being arrogant and aggressive."
#}

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
    #
    # # Perform POS tagging
    # tagged_tokens = pos_tag(tokens)
    #
    # # Lemmatize text
    # lemmatizer = WordNetLemmatizer()
    # lemmatized_tokens = []
    # for token, tag in tagged_tokens:
    #     if tag.startswith('NN'):
    #         pos = 'n'
    #     elif tag.startswith('VB'):
    #         pos = 'v'
    #     else:
    #         pos = 'a'
    #     lemmatized_tokens.append(lemmatizer.lemmatize(token, pos))
    #
    # # Join cleaned tokens
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
print("Phase 1: Combining CSV Data Files")
dfs = []
for value in categories:
    df = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/'+value+'.csv')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


############################## Labeled Data ####################################
print("Phase 2: Extracting Labeled Data")
df_labeled = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/WIP EDI Examples.csv')
# print("BEFORE df_labeled: " + str(len(df_labeled)))
# df_labeled = df_labeled.drop_duplicates(subset=['Text'])
# print("AFTER df_labeled: " + str(len(df_labeled)))

# Encode the labels using LabelEncoder
# label_encoder = LabelEncoder()
# df_labeled['WSDE Label'] = label_encoder.fit_transform(df_labeled['WSDE Label'])

############################# Definition Embedding #############################
print("Phase 3: Fine-Tuning Model")
word_embedding_model = SentenceTransformer('all-mpnet-base-v2')
# word_embedding_model = SentenceTransformer('bert-base-uncased')
pooling_model = models.Pooling(word_embedding_model.get_sentence_embedding_dimension(), pooling_mode="max")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print("Phase TEST: Definition Embedding")
wsde_definition_embeddings = []
for key, value in wsde_definition.items():
    wsde_definition_embeddings.append(model.encode(value, convert_to_tensor=True))

############################ Fine-Tune Model ####################################
print("Phase 3: Fine-Tuning Model")
# word_embedding_model = SentenceTransformer('all-mpnet-base-v2')
# # word_embedding_model = SentenceTransformer('bert-base-uncased')
# pooling_model = models.Pooling(word_embedding_model.get_sentence_embedding_dimension(), pooling_mode="max")
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
#                                           add_special_tokens=True,
#                                           truncation=True,
#                                           padding='max_length',
#                                           max_length=1300,
#                                           return_attention_mask=True,
#                                           return_tensors='pt')

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

# train_texts, val_texts, train_labels, val_labels = train_test_split(sentences1, labels, test_size=0.2, random_state=42)
# train_encoded = tokenizer(train_texts)
# wsde_encoded=tokenizer(sentences2)
# val_encoded = tokenizer(val_texts)
#
# padded_train_input_ids = pad_sequence([torch.tensor(ids) for ids in train_encoded['input_ids']], batch_first=True, padding_value=tokenizer.pad_token_id)
# padded_train_attention_mask = pad_sequence([torch.tensor(mask) for mask in train_encoded['attention_mask']], batch_first=True, padding_value=0)
#
# train_dataset = TensorDataset(padded_train_input_ids,
#                               padded_train_attention_mask,
#                               train_labels)
#
# padded_wsde_input_ids = pad_sequence([torch.tensor(ids) for ids in train_encoded['input_ids']], batch_first=True, padding_value=tokenizer.pad_token_id)
# padded_wsde_attention_mask = pad_sequence([torch.tensor(mask) for mask in train_encoded['attention_mask']], batch_first=True, padding_value=0)
#
# wsde_dataset = TensorDataset(padded_wsde_input_ids,
#                               padded_wsde_attention_mask,
#                               train_labels)
#
# padded_val_input_ids = pad_sequence([torch.tensor(ids) for ids in val_encoded['input_ids']], batch_first=True, padding_value=tokenizer.pad_token_id)
# padded_val_attention_mask = pad_sequence([torch.tensor(mask) for mask in val_encoded['attention_mask']], batch_first=True, padding_value=0)
#
# val_dataset = TensorDataset(padded_val_input_ids, padded_val_attention_mask, val_labels)
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# wsde_loader = DataLoader(wsde_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
#
# # train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32) #, collate_fn=custom_collate)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
# optimizer = optim.Adam(model.parameters(), lr=0.05)
#
# wsde_batch_encoded = []
# for batch in wsde_loader:
#     input_ids, attention_mask, labels = batch
#     format_batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
#     format_labels = {'labels': labels}
#     wsde_batch_encoded.append(format_batch)
#     break
#
# best_val_f1 = 0.0
# best_epoch = 0
# for epoch in range(10):  # Adjust the number of epochs as needed
#     model.train()
#     for batch in train_loader:
#         input_ids, attention_mask, labels = batch
#         optimizer.zero_grad()
#         format_batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
#         format_labels = {'labels': labels}
#
#         # max_seq_length = 512  # Maximum sequence length supported by the model
#         # segments = []
#         # for i in range(0, input_ids.shape[1], max_seq_length):
#         #     segment = input_ids[:, i:i + max_seq_length]
#         #     segments.append(segment)
#         #
#         # # Process each segment separately and aggregate the outputs
#         # aggregated_outputs = []
#         # for segment in segments:
#         #     print(segment)
#         #     outputs = model(segment)
#         #     aggregated_outputs.append(outputs)
#         #
#         # # Combine the outputs from all segments using an aggregation strategy (e.g., averaging, concatenation)
#         # combined_output = torch.mean(torch.stack(aggregated_outputs), dim=0)
#
#         outputs = model(format_batch)
#         wsde_outputs = model(wsde_batch_encoded[0])
#
#         # numerical_outputs = []
#         # for value in outputs.values():
#         #     # Check if the value is a tensor (numerical data)
#         #     if torch.is_tensor(value):
#         #         numerical_outputs.append(value)
#
#         # definitions_embeddings = definitions_embeddings[0].unsqueeze(0)
#         # definitions_embeddings = definitions_embeddings.repeat(16, 1)
#         #
#         # cosine_similarity_scores = cosine_similarity(
#         #     outputs['sentence_embedding'].detach().numpy(),
#         #     definitions_embeddings.detach().numpy())
#         #
#         # cosine_similarity_scores = torch.tensor(cosine_similarity_scores)
#         #
#         # print(cosine_similarity_scores.shape)
#         #
#         # cs_fv = torch.index_select(cosine_similarity_scores, dim=1, index=torch.tensor([0]))
#         # cosine_similarity_scores = cs_fv.squeeze()
#         #
#         # print(cosine_similarity_scores)
#         # print(labels)
#
#         # test_list = []
#         # for values in outputs.values():
#         #     test_list.append(values)
#
#         print(outputs["sentence_embedding"])
#         print(wsde_outputs)
#
#         loss_cs = cosine_similarity(outputs["sentence_embedding"].detach().numpy(),
#                                     wsde_outputs["sentence_embedding"].detach().numpy())
#         print("IS IT REAL? :o")
#         loss_cs.backward()
#         optimizer.step()
#
#     # Validation
#     model.eval()
#     val_predictions = []
#     val_targets = []
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids, attention_mask, labels = batch
#             outputs = model(input_ids, attention_mask=attention_mask)
#             val_predictions.extend(outputs.cpu().numpy())
#             val_targets.extend(labels.cpu().numpy())
#
#     val_predictions = torch.tensor(val_predictions)
#     val_targets = torch.tensor(val_targets)
#     val_f1 = f1_score(val_targets, val_predictions, average='weighted')
#     print(f"Epoch {epoch + 1}/{10}, Validation F1-score: {val_f1}")
#
#     # Check for early stopping
#     if val_f1 > best_val_f1:
#         best_val_f1 = val_f1
#         best_epoch = epoch
#     else:
#         if epoch - best_epoch >= 2:  # Patience: wait for 2 epochs without improvement
#             print("Early stopping.")
#             break
#
# print("Training finished.")

#index_best_epoch = np.argmin(loss_values)
# concat_train_validation_dataloader = ConcatDataset([train_loader.dataset, val_loader.dataset])
# total_dataloader = DataLoader(concat_train_validation_dataloader, batch_size=32, shuffle=True)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=3,
    evaluator=evaluator,
    evaluation_steps=5
)

# train_df, val_df = train_test_split(df_labeled, test_size=0.2, random_state=42)
#
# train_embeddings = model.encode(train_df['Text'].apply(clean_text).values, convert_to_tensor=True)
# train_labels = train_df['WSDE Label'].values
#
# val_embeddings = model.encode(val_df['Text'].apply(clean_text).values, convert_to_tensor=True)
# val_labels = val_df['WSDE Label'].values
#
# print(train_embeddings.shape)
# print(train_labels.type)
#
# model.fit(train_embeddings, train_labels)

# best_f1_score = 0
# best_model = None
#
# optimizer = AdamW(model.parameters(), lr=2e-5)
#
# print("Phase 3a: Fine-Tuning Loop")
# for epoch in range(5):
#     model.train()
#     # train_loader = DataLoader(list(zip(train_embeddings, train_labels)), shuffle=True, batch_size=16)
#     for batch in train_loader:
#         optimizer.zero_grad()
#         embeddings, labels = batch
#         outputs = model.forward(embeddings)
#         loss = model.module.compute_loss(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     # Evaluation on validation data
#     model.eval()
#     with torch.no_grad():
#         val_outputs = model.forward(val_embeddings)
#         val_predictions = torch.argmax(val_outputs, dim=1).cpu().numpy()
#         val_f1_score = f1_score(val_labels, val_predictions, average='weighted')
#
#     print(f"Epoch {epoch + 1}/5 - Validation F1-Score: {val_f1_score}")
#
#     # Keep track of the best performing model
#     if val_f1_score > best_f1_score:
#         best_f1_score = val_f1_score
#         best_model = model
#
# # Best model selection
# model = best_model

# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=3,
#     warmup_steps=3,
#     evaluator=evaluator,
#     evaluation_steps=3
# )

############################# Definition Embedding #############################
print("Phase 4: Definition Embedding")
definitions_embeddings = []
for key, value in wsde_definition.items():
    definitions_embeddings.append(model.encode(value, convert_to_tensor=True))

############################# Data Embedding #############################
print("Phase 5: Data Embedding")

df = df.drop_duplicates(subset=['SubReddit', 'ID'])
indices_to_drop = df[df.apply(lambda row: (row['SubReddit'], row['ID']) in zip(df_labeled['SubReddit'], df_labeled['ID']), axis=1)].index
df.drop(indices_to_drop, inplace=True)

df['Cleaned Text'] = df['Text'].apply(clean_text)

# Extract the cleaned text column
text_data = df['Cleaned Text'].tolist()

# Generate sentence embeddings for text data
data_embeddings = model.encode(text_data, convert_to_tensor=True)

############################# Distances to Category #############################
print("Phase 6: Calculate Data Cosine Distance to the Category")
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

df['Potential Category by SS'] = closest_definitions
df['Potential Category - Cosine Similarity'] = closest_definition_scores

plt.hist(1-df['Potential Category - Cosine Similarity'], bins=20, edgecolor='black')
plt.xlabel('Cosine Distances')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Distances')
plt.show()

model.save("WSDE_model")

df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/'
          'SE4AI - WSDE FT Output.csv', index=False)