import csv
import re
import codecs

import pyLDAvis
import textstat as textstat
from gensim import corpora, models
from gensim.models import Word2Vec, CoherenceModel
from nltk import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')

MG_keywords = ['moralism', 'puritanism', 'prudery', 'morality', 'prudishness', 'effigies', 'dummy', 'puppet', 'likeness', 'statue', 'scapegoats', 'victims', 'excuses', 'pushover', 'stooge', 'sucker', 'sacrifice', 'patsy', 'weakling',
                                 'doormat', 'leadership', 'management', 'governance', 'leaders', 'supervisors', 'chiefs', 'directors', 'guidance', 'lead', 'authority', 'dominion', 'control', 'sway', 'command', 'dominance', 'police', 'prerogative', 'force',
                                 'jurisdiction', 'rule', 'influence', 'leverage', 'clout', 'important', 'money', 'cash', 'coin', 'funds', 'dollar', 'wealth', 'wage', 'salary', 'power', 'energy', 'strength', 'capability', 'ability', 'social status', 'prestige',
                                 'fame', 'dignity', 'esteem', 'importance', 'prominence', 'stature', 'status', 'renown', 'notoriety', 'significance', 'rank', 'level', 'position', 'echelon', 'pride', 'pridefulness', 'confidence', 'ego', 'self-respect', 'honor',
                                 'congratulate', 'reputation', 'repute', 'credit', 'character', 'rule-breaker', 'damsel in distress', 'housewife', 'housewives', 'homemaking', 'homemaker', 'womanly duty', 'womanly duties', 'girly girl', 'girly-girl',
                                 'hoyden', 'ladette', 'lady of the house', 'woman thing', 'catcalling', 'jeering', 'hooting', 'snorting', 'sniffing', 'jibing', 'gibing', 'sneering', 'laughing', 'whistle', 'heckling', 'holler', 'trolling', 'mansplaining',
                                 'lashing out', 'wishful thinking', 'willful denial', 'accused', 'indicted', 'charged', 'blamed', 'prosecuted', 'censured', 'impugned', 'criticized', 'denounced', 'appealed', 'castigated', 'reprobated', 'convicted', 'guilty',
                                 'culpable', 'punishable', 'corrected', 'rectified', 'amended', 'revised', 'culprit', 'imprison', 'rebuke', 'discipline', 'reprimand', 'chide', 'admonish', 'assessed', 'diminished', 'belittle', 'scorn', 'devalue', 'denigrate', 'decry',
                                 'deprecate', 'depreciate', 'derogate', 'outperformed', 'beat', 'exceed', 'surpass', 'outdo', 'defeated', 'bested', 'underestimate', 'underrate', 'undervalue', 'minimize', 'victim blaming', 'punished', 'penalized', 'fined', 'sentenced',
                                 'chastised', 'levied', 'rejected', 'abandoned', 'deserted', 'disused', 'denied', 'disregarded', 'dumped', 'ditched', 'rebuff', 'abrasive', 'irritating', 'annoying', 'harsh', 'bitter', 'cruel', 'unpleasant', 'rough', 'unkind', 'frustrating',
                                 'disturbing', 'aggravating', 'bothersome', 'threatening', 'menacing', 'ominous', 'intimidatory', 'terrorizing', 'sinister', 'powerful woman', 'powerful women', 'less competent', 'incompetent', 'unskillful', 'helpless',
                                 'inadequate', 'incapable', 'unqualified', 'useless', 'inept', 'unfit', 'inexperienced', 'duplicitous', 'deceptive', 'deceitful', 'dishonest', 'fraudulent', 'shady', 'crooked', 'crazy', 'kooky', 'mad', 'nuts', 'nutty', 'silly', 'wacky',
                                 'ridiculous', 'absurd', 'foolish', 'ludicrous', 'mental', 'irrational', 'hysterical', 'agitated', 'distraught', 'frantic', 'frenzied', 'neurotic', 'convulsive', 'upset', 'vindictive', 'malicious', 'vengeful', 'vicious', 'revengeful',
                                 'petty', 'spiteful', 'merciless', 'resentful', 'manipulativeness', 'manipulative', 'exploit', 'deceive', 'devious', 'shrewd', 'coldness', 'detachment', 'objectivity', 'cold', 'frigidness', 'aggression', 'hostility', 'defiance',
                                 'belligerence', 'malice', 'antagonism', 'antipathy', 'malevolence', 'pugnacity', 'encroachment', 'ladylike', 'womanlike']

sentiment_analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Load CSV file, perform sentiment analysis with NLTK stopwords,
# and append results as new columns
def clean_text(text):
    # Remove emojis using regular expression
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # Emoticons
                               "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               "\U0001F700-\U0001F77F"  # Alchemical Symbols
                               "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               "\U0001FA00-\U0001FA6F"  # Chess Symbols
                               "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               "\U00002702-\U000027B0"  # Dingbats
                               "\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)

    # Remove special characters using regex
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = emoji_pattern.sub(r'', cleaned_text)

    return cleaned_text


def analyze_text_and_append(csv_file, output_csv_file):
    stop_words = set(stopwords.words("english"))

    with open(csv_file, 'r', encoding='utf-8', errors='replace') as infile, \
            open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        #TODO: input headers
        fieldnames = reader.fieldnames + ['Filtered Text', 'Word2Vec Embeddings', 'Sentiment', 'Subjectivity', 'Readability']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        lemmatized_documents = []
        counter = 1
        for row in reader:
            text = row['content']  # Extract the 'Text' column value

            #########~~~ Part 1: Remove Stopwords and Clean Text ~~~#########
            # Clean the text from emojis and special characters
            cleaned_text = clean_text(text)

            # Tokenize the cleaned text and remove stopwords
            tokens = [word for word in TextBlob(cleaned_text).words if word.lower() not in stop_words]

            #########~~~ Part 2: Word2Vec ~~~#########
            word2vec_model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, workers=4)

            embeddings = {}
            for word in MG_keywords:
                if word in word2vec_model.wv:
                    embeddings[word] = word2vec_model.wv[word]

            row['Word2Vec Embeddings'] = embeddings

            #########~~~ Part 3: Sentiment Analysis ~~~#########
            filtered_text = " ".join(tokens)
            row['Filtered Text'] = filtered_text

            scores = sentiment_analyzer.polarity_scores(cleaned_text)
            if scores['compound'] > 0:
                row['Subjectivity'] = 'Subjective'
                if scores['pos'] > scores['neg']:
                    row['Sentiment'] = "Positive"
                else:
                    row['Sentiment'] = "Negative"
            else:
                row['Subjectivity'] = 'Objective'

            # Perform sentiment analysis on the filtered tokens
            # filtered_text = " ".join(tokens)
            # analysis = TextBlob(filtered_text)
            # sentiment = "Positive" if analysis.sentiment.polarity > 0 else "Negative"
            # subjectivity = "Subjective" if analysis.sentiment.subjectivity > 0.5 else "Objective"

            # row['Filtered Text'] = filtered_text
            # row['Sentiment'] = sentiment
            # row['Subjectivity'] = subjectivity

            #########~~~ Part 4: Readability ~~~#########
            row['Readability'] = textstat.flesch_reading_ease(text)

            #########~~~ Part 5a: Lemmonize Text ~~~#########

            lemmatized_text = [lemmatizer.lemmatize(word, pos='v') for word in text if word.isalpha()]
            lemmatized_documents.append(lemmatized_text)

            writer.writerow(row)

            print("Completed row " + str(counter))
            counter += 1

    #########~~~ Part 5b: LDA ~~~#########
    # Create a dictionary from the lemmatized documents
    dictionary = corpora.Dictionary(lemmatized_documents)

    # Create a bag-of-words representation of each document
    corpus = [dictionary.doc2bow(document) for document in lemmatized_documents]

    # Train an LDA model on the corpus
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4)
    print(lda_model.print_topics())

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=lemmatized_documents, dictionary=MG_keywords, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, MG_keywords)
    vis

csv_file = 'C:/Users/amand/OneDrive/Desktop/Thesis/Master-Thesis/MG_hot_top_2000.csv'
output_file= 'C:/Users/amand/OneDrive/Desktop/Thesis/Master-Thesis/combo_approach_hot_top_2000.csv'
analyze_text_and_append(csv_file, output_file)
