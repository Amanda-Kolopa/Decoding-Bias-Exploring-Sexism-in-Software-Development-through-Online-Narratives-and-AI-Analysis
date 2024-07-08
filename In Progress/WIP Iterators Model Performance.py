import re

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.stats import ttest_ind, ttest_1samp, mannwhitneyu
from scipy.stats import ttest_rel
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import permutation_test_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from sbert_sklearn_wrapper import SklearnSentenceTransformer


class MockHuman(BaseEstimator, TransformerMixin):
    np.random.seed(42)
    # def __init__(self):
    #     # self.predictions = predictions
    #     print("")

    def fit(self, X, y=None):
        # MockHuman does not need to fit to data, as it's a mock representation
        # self.predictions = X
        return self

    def predict(self, X):
        # Check is fit had been called
        # check_is_fitted(self)

        # Assuming X is not actually used, and predictions are predefined
        return X

    def score(self, X, y):
        # Check is fit had been called
        check_is_fitted(self)

        # Calculate micro F1 score based on predefined predictions and true labels y
        return f1_score(y, self.predict(X), average='micro')

    def set_predictions(self, predictions):
        self.predictions = predictions

    def transform(self, X):
        # Assuming X is not actually used, and predictions are predefined
        return self.predictions

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


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


def map_labels_to_numeric(predictions, label_map):
    return [label_map[label] for label in predictions]


# Define the data
data = {
    'amanda_predictions': [
        ['TI', 'SDP', 'TI', 'GSP', 'FCGS', 'GSP', 'SDP', 'FCGS', 'TI', 'FCGS',
         'SDP', 'TI', 'GSP', 'SDP', 'TI', 'FCGS', 'GSP', 'SDP', 'GSP', 'FCGS']
    ],
    'sharon_predictions': [
        ['FCGS', 'GSP', 'SDP', 'GSP', 'TI', 'SDP', 'FCGS', 'GSP', 'FCGS', 'FCGS',
         'TI', 'TI', 'TI', 'FCGS', 'FCGS', 'FCGS', 'TI', 'SDP', 'FCGS', 'FCGS'],
        ['FCGS', 'GSP', 'SDP', 'SDP', 'TI', 'SDP', 'FCGS', 'GSP', 'FCGS', 'FCGS',
         'TI', 'TI', 'TI', 'FCGS', 'FCGS', 'FCGS', 'TI', 'SDP', 'FCGS', 'FCGS'],
    ],
    'anick_predictions': [
        ['FCGS', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'TI', 'FCGS', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'FCGS', 'TI', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'FCGS', 'FCGS', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],

        ['FCGS', 'SDP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'SDP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'TI', 'FCGS', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'SDP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'FCGS', 'TI', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'SDP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'FCGS', 'FCGS', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],

        ['FCGS', 'GSP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'TI', 'TI', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'GSP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'TI', 'FCGS', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'GSP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'FCGS', 'TI', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS'],
        ['FCGS', 'GSP', 'TI', 'GSP', 'GSP', 'TI', 'TI', 'GSP', 'TI', 'FCGS',
         'GSP', 'TI', 'TI', 'FCGS', 'FCGS', 'GSP', 'GSP', 'TI', 'GSP', 'FCGS']
    ],
    'ezgi_predictions': [
        ['FCGS', 'GSP', 'TI', 'SDP', 'FCGS', 'GSP', 'FCGS', 'GSP', 'TI', 'SDP',
         'SDP', 'GSP', 'TI', 'FCGS', 'FCGS', 'OI', 'GSP', 'SDP', 'GSP', 'FCGS']
    ],
    'true_value': [
        ['FCGS', 'GSP', 'TI', 'GSP', 'FCGS', 'GSP', 'FCGS', 'GSP', 'TI', 'FCGS',
         'SDP', 'TI', 'TI', 'FCGS', 'FCGS', 'FCGS', 'GSP', 'SDP', 'GSP', 'FCGS']
    ]
}

label_mapping = {'FCGS': 1, 'GSP': 2, 'TI': 3, 'SDP': 4, 'OI': 5}

############################## Iterator Evaluation ####################################

# Calculate F1-scores and other metrics for each predictor
predictors = ['amanda', 'sharon', 'anick', 'ezgi']
results = {}
reference_f1_scores = []

# Calculate reference F1 scores
# for predictor in predictors:
#     predictor_predictions = data[f'{predictor}_predictions']
#     predictor_f1_scores = [f1_score(pred, data['true_value'][0], average='micro') for pred in predictor_predictions]
#     reference_f1_scores.extend(predictor_f1_scores)
#
# reference_avg_f1_score = np.mean(reference_f1_scores)
# print(reference_avg_f1_score)

# for predictor in predictors:
#     predictor_predictions = data[f'{predictor}_predictions']
#     true_value = data['true_value'][0]
#
#     # Calculate F1 scores for each set of predictions
#     predictor_f1_scores = [f1_score(pred, true_value, average='micro') for pred in predictor_predictions]
#     predictor_avg_f1_score = np.mean(predictor_f1_scores)
#     predictor_std_dev_f1_score = np.std(predictor_f1_scores)
#
#     # Map predictions to corresponding values
#     combined_predictions = [label_mapping[pred] for sublist in predictor_predictions for pred in sublist]
#     true_value_mapped = [label_mapping[label] for label in true_value]
#
#     # Calculate p-value using paired t-test
#     _, p_value = ttest_rel(combined_predictions, true_value_mapped)
#
#     # Store results for the predictor
#     results[predictor] = {
#         'Average F1-score': predictor_avg_f1_score,
#         'Standard Deviation F1-score': predictor_std_dev_f1_score,
#         'P-value': p_value
#     }

np.random.seed(42)

for predictor in predictors:
    predictor_predictions = data[f'{predictor}_predictions']  # Extract inner list of predictions
    true_value = data['true_value'][0]

    if len(predictor_predictions) == 1:
        # Calculate F1 scores for each set of predictions
        predictor_f1_scores = [f1_score(pred, true_value, average='micro') for pred in predictor_predictions]
        predictor_avg_f1_score = np.mean(predictor_f1_scores)
        predictor_std_dev_f1_score = np.std(predictor_f1_scores)

        # Map predictions to corresponding values
        combined_predictions = [label_mapping[pred] for sublist in predictor_predictions for pred in sublist]
        true_value_mapped = [label_mapping[label] for label in true_value]

        # Store results for the predictor
        results[predictor] = {
            'Average F1-score': predictor_avg_f1_score,
            'Standard Deviation F1-score': predictor_std_dev_f1_score
        }
    else:
        avg_f1_scores = []
        std_dev_f1_scores = []

        for inner_predictions in predictor_predictions:
            # Calculate F1 score for each set of predictions
            predictor_f1_score = f1_score(inner_predictions, true_value, average='micro')

            # Map predictions to corresponding values
            combined_predictions = [label_mapping[pred] for pred in inner_predictions]
            true_value_mapped = [label_mapping[label] for label in true_value]

            avg_f1_scores.append(predictor_f1_score)

        # Calculate average F1 score, standard deviation of F1 scores, and average p-value
        predictor_avg_f1_score = np.mean(avg_f1_scores)
        predictor_std_dev_f1_score = np.std(avg_f1_scores)

        results[predictor] = {
            'Average F1-score': predictor_avg_f1_score,
            'Standard Deviation F1-score': predictor_std_dev_f1_score
        }

# Prepare data for use with the mock model
X = {}
y = []

# Add data from each predictor to X and true_value to y
for predictor_key, predictions_list in data.items():
    if predictor_key.endswith('_predictions'):
        predictor_name = predictor_key.split('_')[0]  # Extract predictor name
        if predictor_name not in X:
            X[predictor_name] = []
        X[predictor_name].extend(predictions_list)

    elif predictor_key == 'true_value':
        y.extend(data['true_value'][0])  # Assuming true_value is consistent across all instances

y = np.array(y)

# Create predictions dictionary for the mock model
predictions = {}
for predictor_name, predictions_list in X.items():
    predictions[predictor_name] = predictions_list

# Initialize Logistic Regression model
clf = LogisticRegression(max_iter=50000)
# Custom scorer for F1-score
f1_scorer = make_scorer(f1_score, average='micro')

avg_p_value = {}
# Perform permutation test score for each predictor separately
for predictor_name, predictor_data in X.items():
    if predictor_name == 'true_value':
        continue  # Skip true_value, we only need predictor data

    avg_p_value[predictor_name] = []

    for inner_data in predictor_data:
        test_inner_data = np.array(inner_data)
        X_numerical = np.array([label_mapping[val] for val in test_inner_data]).reshape(-1, 1)
        y_numerical = np.array([label_mapping[val] for val in data['true_value'][0]])

        assert X_numerical.shape[0] == y_numerical.shape[0], \
            f"Number of samples in X_numerical ({X_numerical.shape[0]}) does not match y_numerical ({y_numerical.shape[0]})"

        mock_model = MockHuman()
        # pipeline_human = make_pipeline(mock_model, clf)

        score, perm_scores, pvalue = permutation_test_score(
            mock_model, X_numerical, y_numerical, cv=2,
            scoring=f1_scorer, n_permutations=100, n_jobs=2
        )


        # predictions = {predictor_name: data}
        # mock_model = MockHuman()
        # mock_model.setPredictions(predictions)
        # pipeline_human = make_pipeline(mock_model, clf)
        # score, perm_scores, pvalue = permutation_test_score(
        #     mock_model, data, y, cv=2, scoring=f1_scorer, n_permutations=100, n_jobs=2
        # )
        avg_p_value[predictor_name].append(pvalue)
    # Append p-value to results for the predictor name
    #results[predictor_name]['p_value'] = pvalue
    print(avg_p_value)
    results[predictor_name]['P-value'] = np.mean(avg_p_value[predictor_name])


# Print results
# for predictor, result in results.items():
#     print(f"Predictor: {predictor}")
#     print(result)
#     print(f"Average F1-score: {result['Average F1-score']}")
#     print(f"Standard Deviation: {result['Standard Deviation F1-score']}")
#     print(f"P-value: {result['P-value']}")
#     print()


############################## Model Evaluation ####################################

df_labeled = pd.read_csv(
    'C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/WIP Full Text EDI Classification.csv',
    encoding="iso-8859-1")

X = df_labeled['Text'].apply(clean_text).to_numpy()
y = df_labeled['Taxonomy Label'].to_numpy()

assert len(X) == len(y), "The number of sentences and labels must be equal."

sbert = SklearnSentenceTransformer('taxonomy_model_FULL_TEXT')
# clf = LogisticRegression(max_iter=1000)  # Ensure enough iterations for convergence
f1_scorer = make_scorer(f1_score, average='micro')
# pipeline = make_pipeline(sbert, clf)

# Perform permutation test score
score, perm_scores, pvalue = permutation_test_score(
    sbert, X, y, cv=2, scoring=f1_scorer, n_permutations=100, n_jobs=2
)

# Update model results
results['model'] = {
    'Average F1-score': score,
    'Standard Deviation F1-score': np.std(perm_scores),
    'P-value': pvalue
}

for predictor, result in results.items():
    print(f"Predictor: {predictor}")
    print(result)
    print(f"Average F1-score: {result['Average F1-score']}")
    print(f"Standard Deviation: {result['Standard Deviation F1-score']}")
    print(f"P-value: {result['P-value']}")
    print()

# ############################## Model Diagram (F1-Score with S.D) ####################################
predictor_names = list(results.keys())
avg_f1_scores = [result['Average F1-score'] for result in results.values()]
std_dev_f1_scores = [result['Standard Deviation F1-score'] for result in results.values()]
p_values = [result['P-value'] for result in results.values()]

# Creating index for x-axis
index = np.arange(len(predictor_names))

# Plotting bar chart
bars = plt.bar(index, avg_f1_scores, color='#ADD8E6', alpha=0.7, label='Average F1-Score')

# Adding error bars
for bar, std_dev in zip(bars, std_dev_f1_scores):
    height = bar.get_height()
    plt.errorbar(bar.get_x() + bar.get_width() / 2, height, yerr=std_dev, fmt='none', ecolor='black', capsize=5)

# Adding labels and title
plt.xlabel('Predictor')
plt.ylabel('Average F1-Score')
plt.title('Average F1-Scores with Standard Deviations per Predictor')
plt.xticks(index, predictor_names, rotation=45, ha='right')

# Adding text labels for F1-score, standard deviation, and p-value
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2,
             0.1,
             f'{avg_f1_scores[i]:.2f} ± {std_dev_f1_scores[i]:.2f}',
             ha='center')

    # Adding asterisks based on p-values
    p_value = p_values[i]
    if p_value < 0.001:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 0.2,
                 '***',
                 ha='center', color='red', fontsize=10)
    elif p_value < 0.01:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 0.2,
                 '**',
                 ha='center', color='red', fontsize=10)
    elif p_value < 0.3:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 0.2,
                 '*',
                 ha='center', color='red', fontsize=10)

# Adding a bar for chance level (for example, 0.5)
plt.axhline(y=0.5, color='r', linestyle='--', label='Chance')

# Display the plot
plt.legend()
plt.tight_layout()
plt.show()

# ############################## Model Diagram (p-values) ####################################
# Extracting predictor names and p-values
# predictor_names = list(results.keys())
# p_values = [result.get('P-value', np.nan) for result in results.values()]
#
# # Plotting bar chart
# bars = plt.bar(predictor_names, p_values, color='skyblue', alpha=0.7)
#
# # Adding labels and title
# plt.xlabel('Predictor/Model')
# plt.ylabel('P-value')
# plt.title('P-values for Predictors and Model')
#
# # Rotating x-axis labels for better readability
# plt.xticks(rotation=45, ha='right')
#
# # Adding text labels for F1-score and standard deviation
# for i, bar in enumerate(bars):
#     plt.text(bar.get_x() + bar.get_width() / 2,
#              bar.get_height(),
#              f'p={p_values[i]:.2f}',
#              ha='center')
#
# # Display the plot
# plt.tight_layout()
# plt.show()


# ############################## Save Results ####################################

# Create a DataFrame to store the results
results_df = pd.DataFrame(results).T

# Save the DataFrame to a CSV file
results_df.to_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/In Progress/Performance Results.csv',
                  index=False)
