import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text, plot_tree
# TODO: after preliminary analysis, incorporate proper train/test split

# Load training data from CSV file
experiment_data = pd.read_csv('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/Model Scripts/Topic Modeling for Filtration/Experimental - NEW TRAIN LDA - TI.csv')

feature_groups = {
    'Topic Model with Post Title': ['Top Topic using Post Title - Word 1',
                                    'Top Topic using Post Title - Word 2',
                                    'Top Topic using Post Title - Word 3',
                                    'Top Topic using Post Title - Word 4',
                                    'Top Topic using Post Title - Word 5',
                                    'Probability using Post Title'],
    'Topic Model with Vocabulary': ['Top Topic 1 using Vocabulary',
                                    'Probability 1 using Vocabulary',
                                    'Top Topic 2 using Vocabulary',
                                    'Probability 2 using Vocabulary',
                                    'Top Topic 3 using Vocabulary',
                                    'Probability 3 using Vocabulary'],
    'Topic Model with Dictionary': ['Top Topic 1 using Dataset',
                                    'Probability 1 using Dataset',
                                    'Top Topic 2 using Dataset',
                                    'Probability 2 using Dataset',
                                    'Top Topic 3 using Dataset',
                                    'Probability 3 using Dataset']
}

# Assuming the last column is the target variable
X_train = experiment_data[['Top Topic using Post Title - Word 1'] +
                          ['Top Topic using Post Title - Word 2'] +
                          ['Top Topic using Post Title - Word 3'] +
                          ['Top Topic using Post Title - Word 4'] +
                          ['Top Topic using Post Title - Word 5'] +
                          ['Probability using Post Title'] +
                          ['Top Topic 1 using Vocabulary'] +
                          ['Probability 1 using Vocabulary'] +
                          ['Top Topic 2 using Vocabulary'] +
                          ['Probability 2 using Vocabulary'] +
                          ['Top Topic 3 using Vocabulary'] +
                          ['Probability 3 using Vocabulary'] +
                          ['Top Topic 1 using Dataset'] +
                          ['Probability 1 using Dataset'] +
                          ['Top Topic 2 using Dataset'] +
                          ['Probability 2 using Dataset'] +
                          ['Top Topic 3 using Dataset'] +
                          ['Probability 3 using Dataset']
]

# One-hot encoding for String column data
X_train = pd.get_dummies(X_train, columns=['Top Topic using Post Title - Word 1',
                                           'Top Topic using Post Title - Word 2',
                                           'Top Topic using Post Title - Word 3',
                                           'Top Topic using Post Title - Word 4',
                                           'Top Topic using Post Title - Word 5',
                                           'Top Topic 1 using Vocabulary',
                                           'Top Topic 2 using Vocabulary',
                                           'Top Topic 3 using Vocabulary',
                                           'Top Topic 1 using Dataset',
                                           'Top Topic 2 using Dataset',
                                           'Top Topic 3 using Dataset'
                                           ])
print(X_train)

# X_train = experiment_data.iloc[:, :-1]
y_train = experiment_data['Target']

# Split the data into training and testing sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train RandomForest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_split, y_train_split)

# Find the tree with the highest accuracy on the training set
best_tree_index = None
best_tree_accuracy = 0.0

for i, tree in enumerate(random_forest.estimators_):
    accuracy = tree.score(X_train, y_train)
    if accuracy > best_tree_accuracy:
        best_tree_accuracy = accuracy
        best_tree_index = i

# Output the rules of the best decision tree
best_tree = random_forest.estimators_[best_tree_index]
tree_rules = export_text(best_tree, feature_names=list(X_train.columns))
print(f"Best Decision Tree (index {best_tree_index}):\n{tree_rules}")

# Visualize the best decision tree
plt.figure(figsize=(12, 8))
plot_tree(best_tree, feature_names=list(X_train.columns), class_names=True, filled=True, rounded=True)
plt.savefig('C:/Users/amand/OneDrive/Desktop/Thesis/Updated_Thesis/Model Scripts/Topic Modeling for Filtration/Best Tree - TI.pdf')
plt.show()

# Make predictions on the testing data
y_pred = random_forest.predict(X_test_split)

# Calculate F1-score
f1 = f1_score(y_test_split, y_pred, average='weighted')
print(f"F1-score: {f1}")
