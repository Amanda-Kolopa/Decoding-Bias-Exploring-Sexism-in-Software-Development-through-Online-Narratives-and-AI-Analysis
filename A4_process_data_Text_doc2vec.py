import re

import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from gap_statistic import OptimalK
from textblob import TextBlob
from transformers import pipeline
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

categories = {
    'feminine-coded goods and services': [
        'cool', 'natural', 'healthy', 'loyal', 'good', 'affection', 'adoration', 'indulgence', 'loving', 'acceptance',
        'nurturing', 'safety', 'security', 'safe haven', 'kindness', 'compassion', 'mortal attention', 'concern',
        'soothing', 'caring', 'trust', 'respect', 'attentive', 'relationship', 'giving', 'awesome', 'wonderful',
        'lovely', 'great', 'excellent', 'beautiful', 'terrific', 'fantastic', 'fabulous', 'superb', 'hot', 'marvelous',
        'stellar', 'fine', 'neat', 'prime', 'heavenly', 'calm', 'genuine', 'unaffected', 'simple', 'honest', 'innocent',
        'naïve', 'sincere', 'pure', 'raw', 'organic', 'wholesome', 'easy', 'strong', 'fit', 'hearty', 'active',
        'lively', 'faithful', 'dependable', 'devoted', 'trustworthy', 'trusty', 'dedicated', 'reliable', 'pleasant',
        'positive', 'favorable', 'valuable', 'noble', 'decent', 'ethical', 'mortal', 'auspicious', 'happy', 'sentiment',
        'liking', 'veneration', 'blessing', 'privilege', 'courtesy', 'leniency', 'permissiveness', 'admiring',
        'affectionate', 'amiable', 'adoring', 'passionate', 'approval', 'support', 'embracing', 'adoption', 'female',
        'feminine', 'matronly', 'womanly', 'parental', 'protection', 'safeguards', 'safeness', 'guard', 'safekeeping',
        'shield', 'goodwill', 'grace', 'kindliness', 'benevolence', 'gentleness', 'sweetness', 'kindheartedness',
        'benignity', 'empathy', 'sympathy', 'mercy', 'pity', 'commiseration', 'worry', 'fear', 'anxiety', 'unease',
        'concernment', 'relaxing', 'comforting', 'tranquilizing', 'calming', 'hypnotic', 'quieting', 'sedative',
        'dreamy', 'peaceful', 'restful', 'reassuring', 'compassionate', 'benevolent', 'helpful', 'sympathetic',
        'thoughtful', 'generous', 'humane', 'kindly', 'warm', 'soft', 'sensitive', 'tender', 'responsive', 'receptive',
        'considerate', 'warmhearted', 'tenderhearted', 'softhearted', 'nice', 'confide', 'confidence', 'faith',
        'assurance', 'entrustment', 'credence', 'depend on', 'count on', 'admiration', 'regard', 'appreciation',
        'praise', 'recognition', 'reverence', 'kind', 'respectful', 'solicitous', 'gracious', 'polite', 'connection',
        'association', 'kinship', 'relation', 'linkage', 'affiliation', 'interaction', 'bond', 'communication',
        'friendship', 'bestowing', 'offering'
    ],
    'testimonial injustice': [
        'catcalling', 'trolling', 'condescending', 'mansplain', 'moralizing', 'blaming', 'silencing', 'lampooning',
        'satirizing', 'sexualizing', 'desexualizing', 'belittling', 'caricaturing', 'exploiting', 'erasing',
        'infantilizing', 'ridiculing', 'humiliating', 'mocking', 'slurring', 'vilifying', 'demonizing', 'shunning',
        'shaming', 'patronizing', 'dismissive', 'disparaging', 'less credible', 'less competent', 'accused', 'impugned',
        'convicted', 'corrected', 'diminished', 'outperformed', 'jeering', 'hooting', 'snorting', 'sniffing', 'jibing',
        'gibing', 'sneering', 'laughing', 'whistle', 'heckling', 'holler', 'bossy', 'impudent', 'snooty', 'lecturing',
        'preaching', 'condemning', 'condemn', 'condemned', 'faulting', 'denouncing', 'knocking', 'attacking',
        'slamming', 'censuring', 'suppressing', 'quelling', 'subduing', 'censor', 'muffling', 'spoofing', 'burlesquing',
        'mimicking', 'banter', 'bitterness', 'cynicism', 'minimizing', 'discounting', 'derogating', 'pejorative',
        'contemptuous', 'contempt', 'deride', 'scoff', 'taunt', 'tease', 'parodying', 'imitating', 'abuse',
        'manipulate', 'misuse', 'eradicating', 'destroying', 'abolishing', 'obliterating', 'immaturity', 'ignorance',
        'childishness', 'derisive', 'baiting', 'deriding', 'fooling', 'mortifying', 'demeaning', 'embarrassing',
        'degrading', 'ignominious', 'humbling', 'uncivil', 'sarcastic', 'satirical', 'disrespectful', 'sardonic',
        'negativistic', 'disgrace', 'insinuate', 'affronting', 'blaspheming', 'cursing', 'berating', 'insulting',
        'offensive', 'rude', 'abusive', 'malign', 'smearing', 'libeling', 'slandering', 'defaming', 'discrediting',
        'diabolize', 'torment', 'affliction', 'avoidance', 'ostracism', 'exile', 'isolation', 'rejection', 'expulsion',
        'evasion', 'disgracing', 'dishonoring', 'abasement', 'mortification', 'deceiving', 'groveling', 'grudging',
        'domineering', 'dominant', 'disdainful', 'authoritarian', 'snobbish', 'dismissing', 'denigrating',
        'bad-mouthing', 'derogative', 'defamatory', 'deprecatory', 'incompetent', 'unskillful', 'helpless',
        'inadequate', 'incapable', 'unqualified', 'useless', 'inept', 'unfit', 'inexperienced', 'indicted', 'charged',
        'blamed', 'prosecuted', 'censured', 'criticized', 'denounced', 'appealed', 'castigated', 'reprobate', 'guilty',
        'culpable', 'punishable', 'rectified', 'amended', 'revised', 'culprit', 'imprison', 'rebuke', 'discipline',
        'reprimand', 'chide', 'admonish', 'assessed', 'scorn', 'devalue', 'denigrate', 'decry', 'deprecate',
        'depreciate', 'derogate', 'beat', 'exceed', 'surpass', 'outdo', 'defeated', 'bested'
    ],
    'gendered split perception': [
        'duplicitous', 'vindictive', 'conniving', 'untrustworthy', 'careless', 'shady', 'crooked', 'rule-breaker',
        'dangerous', 'suspicious', 'risky', 'deceptive', 'deceitful', 'dishonest', 'fraudulent', 'malicious',
        'vengeful', 'vicious', 'revengeful', 'petty', 'spiteful', 'merciless', 'resentful', 'scheming', 'plotting',
        'conspiring', 'collusive', 'shifty', 'disloyal', 'unreliable', 'untrusty', 'devious', 'unfaithful',
        'thoughtless', 'reckless', 'sloppy', 'negligent', 'indifferent', 'unconcerned', 'absent-minded', 'unthinking',
        'cursory', 'inconsiderate', 'unmindful', 'incautious', 'impetuous', 'unwary', 'mindless', 'dubious',
        'questionable', 'unscrupulous', 'dodgy', 'suspect', 'fishy', 'disreputable', 'suborned', 'corrupt',
        'dishonorable', 'troubling', 'perilous', 'precarious', 'ugly', 'unsafe', 'unstable', 'alarming', 'menacing',
        'insecure', 'irresponsible', 'distrustful', 'skeptical', 'mistrustful', 'unusual', 'unbelieving', 'leery',
        'hazardous', 'threatening', 'dicey', 'misleading', 'sneaky', 'spurious', 'ambiguous', 'delusive', 'fallacious',
        'delusory', 'beguiling'
    ],
    'social dominance penalty': [
        'smother', 'intimidate', 'powerful women', 'powerful woman', 'threatening', 'underestimate', 'doubt',
        'victim blaming', 'crazy', 'hysterical', 'disliked', 'rejected', 'hostile', 'abrasive', 'manipulative',
        'arrogant', 'aggressive', 'ballbreaker', 'castrating bitch', 'punished', 'real woman', 'real women', 'bitch',
        'witch', 'unfair', 'rigid', 'cold', 'psychotic', 'overwhelm', 'stifle', 'repress', 'hold back', 'restrain',
        'bottle up', 'bully', 'frighten', 'scare', 'coerce', 'startle', 'browbeat', 'harass', 'bulldoze', 'pressure',
        'terrify', 'hound', 'daunt', 'oppress', 'constrain', 'dishearten', 'dismay', 'ominous', 'intimidatory',
        'terrorizing', 'sinister', 'underrate', 'undervalue', 'minimize', 'disbelief', 'hesitation', 'uncertainty',
        'skepticism', 'kooky', 'mad', 'nuts', 'nutty', 'silly', 'wacky', 'ridiculous', 'absurd', 'foolish', 'ludicrous',
        'mental', 'irrational', 'agitated', 'distraught', 'frantic', 'frenzied', 'neurotic', 'convulsive', 'upset',
        'hatred', 'disgust', 'hostility', 'loath', 'disapproval', 'distaste', 'animosity', 'aversion', 'antagonism',
        'displeasure', 'antipathy', 'enmity', 'animus', 'disinclination', 'repugnance', 'detestation', 'abhor',
        'detest', 'execrated', 'despised', 'abandoned', 'deserted', 'disused', 'denied', 'disregarded', 'dumped',
        'ditched', 'rebuff', 'antagonistic', 'mean', 'hateful', 'inhospitable', 'nasty', 'unfavorable', 'unfriendly',
        'catty', 'sour', 'inimical', 'negative', 'irritating', 'annoying', 'harsh', 'cruel', 'unpleasant', 'rough',
        'unkind', 'frustrating', 'disturbing', 'aggravating', 'bothersome', 'deceive', 'shrewd', 'vain', 'smug',
        'pompous', 'imperious', 'cocky', 'conceited', 'cavalier', 'bumptious', 'assumptive', 'pretentious',
        'belligerent', 'combative', 'destructive', 'intrusive', 'assertive', 'malevolent', 'pushy', 'pugnacious',
        'penalized', 'fined', 'sentenced', 'chastised', 'levied', 'floozy', 'harlot', 'hussy', 'slut', 'tart', 'tramp',
        'vamp', 'wench', 'whore', 'broad', 'hellion', 'termagant', 'vixen', 'hag', 'shrew', 'foul', 'shameful',
        'biased', 'prejudiced', 'discriminatory', 'strict', 'rigorous', 'stern', 'stringent', 'aloof', 'distant',
        'frigid', 'apathetic', 'glacial', 'demented', 'insane', 'unhinged', 'lunatic', 'paranoid', 'psycho', 'maniac'
    ]
}


def clean_text(text):
    # Remove special characters
    # text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    cleaned_text = text.encode('ascii', 'ignore').decode('utf-8')

    # Remove stopwords and lemmatize the text
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenized_text = [word_tokenize(text) for text in cleaned_text.lower()]
    filtered_text = [[lemmatizer.lemmatize(word) for word in doc if word not in stop_words] for doc in tokenized_text]

    # Create tagged documents
    tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(filtered_text)]

    return tagged_documents


# Read CSV file
df = pd.read_csv(
    'C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/testimonial injustice.csv',
    encoding="utf-8")

# Extract the text column
tagged_documents = clean_text(df['Text'])

# Train the model
model = Doc2Vec(tagged_documents, vector_size=100, window=2, min_count=1, workers=4, epochs=100)

# Generate the document embeddings
X = [model.infer_vector(doc.words) for doc in tagged_documents]


############################# Number of Clusters Optimization #############################

# Use the Elbow Method to determine the optimal number of clusters
# sse = {}
# for k in range(2, 15):
#     kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
#     sse[k] = kmeans.inertia_
#
# plt.figure()
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel("Number of clusters")
# plt.ylabel("SSE")
# plt.title("Elbow Method For Optimal k")
# plt.show()
#
# # Use the Silhouette Coefficient Method to determine the optimal number of clusters
# sil = []
# for k in range(2, 15):
#     kmeans = KMeans(n_clusters=k).fit(X)
#     preds = kmeans.fit_predict(X)
#     sil.append(silhouette_score(X, preds, metric='euclidean'))
#
# plt.figure()
# plt.plot(range(2, 15), sil, 'bx-')
# plt.title('Silhouette Method For Optimal k')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.show()
#
# # Calculate Calinski-Harabasz index for each k value
# ch_scores = []
# for k in range(2, 15):
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
#     ch_scores.append(calinski_harabasz_score(X, kmeans.labels_))
#
# plt.figure()
# plt.plot(range(2, 15), ch_scores, 'bx-')
# plt.title('Calinski-Harabasz For Optimal k')
# plt.xlabel('Number of clusters')
# plt.ylabel('Calinski-Harabasz Score')
# plt.show()
#
# # Calculate Davies-Bouldin index for each k value
# db_scores = []
# for k in range(2, 15):
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
#     db_scores.append(davies_bouldin_score(X, kmeans.labels_))
#
# plt.figure()
# plt.plot(range(2, 15), db_scores, 'bx-')
# plt.title('Davies-Bouldin For Optimal k')
# plt.xlabel('Number of clusters')
# plt.ylabel('Davies-Bouldin Score')
# plt.show()


############################# Kmeans + LDA for Top 10 Words per Cluster #############################

# Cluster the documents based on relevant context using KMeans
kmeans = KMeans(n_clusters=9, random_state=0).fit(X)

df['Cluster ID'] = kmeans.labels_

# Use LDA on the defined clusters to obtain the top 10 words per cluster
lda = LatentDirichletAllocation(n_components=9, random_state=0)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Cleaned Text'])
lda.fit(X)
feature_names = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    # print(f"Cluster {i}:")
    # print(", ".join([feature_names[j] for j in topic.argsort()[:-11:-1]]))
    df.loc[df['Cluster ID'] == i, 'LDA Top 10 Words'] = ", ".join([feature_names[j] for j in topic.argsort()[:-11:-1]])

# Create a new csv file called output.csv that follows the same format as input.csv but has 2 additional columns, Cluster ID and LDA Top 10 Words
df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/text_doc2vec_kmeans_optimization.csv', index=False)
