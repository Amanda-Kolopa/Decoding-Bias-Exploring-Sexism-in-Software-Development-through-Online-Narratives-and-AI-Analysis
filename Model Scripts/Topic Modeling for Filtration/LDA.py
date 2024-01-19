###########################################################
## This script is meant to filter the Testimonial Injustice
## data using the primary keywords from the taxonomy and
## topic modeling with word embeddings and LDA
###########################################################

import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
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


# Read CSV file
df = pd.read_csv(
    'C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/cleaned text data files/testimonial injustice.csv',
    encoding="utf-8")

############################# Grouping Based on Post Title #############################

# Create a TfidfVectorizer object to generate word embeddings using keywords!
vectorizer_t = TfidfVectorizer(stop_words='english')

temp_df = df.loc[:, ['Post Title']]
temp_df = temp_df.drop_duplicates(subset=['Post Title']).reset_index(drop=True)

cleaned_text_t = temp_df['Post Title'].apply(clean_text)

# Generate the document-term matrix
Xt = vectorizer_t.fit_transform(cleaned_text_t)

# Apply LDA to extract topics from the text data
lda_t = LatentDirichletAllocation(n_components=6, random_state=42)
lda_t.fit(Xt)

lda_topic_matrix_t = lda_t.transform(Xt)
assigned_topics = lda_topic_matrix_t.argmax(axis=1)

feature_names_t = np.array(vectorizer_t.get_feature_names_out())

columns = ['Post Title', 'Top Topics using Post Title', 'Probability using Post Title']
topics_df = pd.DataFrame(columns=columns)

for i in temp_df.index:
    top_topic_index_t = np.argmax(lda_topic_matrix_t[i])
    top_topic_probability_t = lda_topic_matrix_t[i, top_topic_index_t]
    top_topic_words_t = [feature_names_t[j] for j in lda_t.components_[top_topic_index_t].argsort()[-5:][::-1]]

    row = {'Post Title': temp_df.loc[i, 'Post Title'],
           'Top Topics using Post Title': ', '.join(top_topic_words_t),
           'Probability using Post Title': top_topic_probability_t}
    topics_df = topics_df._append(row, ignore_index=True)

result_df = pd.DataFrame({'Post Title': temp_df['Post Title'], 'Topic Index': assigned_topics + 1})

result_df = result_df.merge(topics_df, on='Post Title')
df = df.merge(result_df, on='Post Title')
del df['Topic Index']
print(df)

############################# Grouping Based on Vocabulary #############################

# Create a TfidfVectorizer object to generate word embeddings using keywords!
vectorizer_v = TfidfVectorizer(stop_words='english', vocabulary=categories['testimonial injustice'])

cleaned_text_v = df['Text'].apply(clean_text)

# Generate the document-term matrix
Xv = vectorizer_v.fit_transform(cleaned_text_v)

# Apply LDA to extract topics from the text data
lda_v = LatentDirichletAllocation(n_components=25, random_state=42)
lda_v.fit(Xv)

lda_topic_matrix_v = lda_v.fit_transform(Xv)

# Step 3: Extract top topic for each document
df['Top Topic using Vocabulary'] = ''
df['Probability using Vocabulary'] = ''

feature_names_v = vectorizer_v.get_feature_names_out()

for i in range(len(df)):
    top_topic_index_v = np.argmax(lda_topic_matrix_v[i])
    top_topic_probability_v = lda_topic_matrix_v[i, top_topic_index_v]

    # Get the top words for the selected topic
    top_topic_words_v = [feature_names_v[j] for j in lda_v.components_[top_topic_index_v].argsort()[-5:][::-1]]

    df.at[i, 'Top Topic using Vocabulary'] = ', '.join(top_topic_words_v)
    df.at[i, 'Probability using Vocabulary'] = top_topic_probability_v

print(df)

############################# Grouping Based on Dataset Corpus #############################

# Create a TfidfVectorizer object to generate word embeddings using keywords!
vectorizer_d = TfidfVectorizer(stop_words='english')

cleaned_text_d = df['Text'].apply(clean_text)

# Generate the document-term matrix
Xd = vectorizer_d.fit_transform(cleaned_text_d)

# Apply LDA to extract topics from the text data
lda_d = LatentDirichletAllocation(n_components=25, random_state=42)
lda_topic_matrix_d = lda_d.fit_transform(Xd)

# Step 3: Extract top topic for each document
df['Top Topic using Dataset'] = ''
df['Probability using Dataset'] = ''

feature_names_d = vectorizer_d.get_feature_names_out()

for i in range(len(df)):
    top_topic_index_d = np.argmax(lda_topic_matrix_d[i])
    top_topic_probability_d = lda_topic_matrix_d[i, top_topic_index_d]

    # Get the top words for the selected topic
    top_topic_words_d = [feature_names_d[j] for j in lda_d.components_[top_topic_index_d].argsort()[-5:][::-1]]

    df.at[i, 'Top Topic using Dataset'] = ', '.join(top_topic_words_d)
    df.at[i, 'Probability using Dataset'] = top_topic_probability_d

print(df)

# Save the merged DataFrame to a new CSV file
df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/Model Scripts/Topic Modeling for Filtration/LDA - TI.csv', index=False)