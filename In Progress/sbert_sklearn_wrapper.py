# sbert_wrapper.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

definitions = {
    '1':
        "The experiences when women software developers are expected to naturally provide to men because they are "
        "entitled to receive the benefits of women’s goods and services. Moreover, these characteristics are used to "
        "reinforce traditional gender roles. For example, care-mongering is when women are disproportionately "
        "required to be caring and are expected to develop personal relationships with individuals. For example, "
        "I am the only woman in our dev team and I am always implicitly expected to do the administrative tasks "
        "during our meetings. When I confront my team about this, they explain that my organization and note-taking "
        "abilities are a natural talent that benefits the team.",
    '2':
        "Women software developers experience harsher judgement when performing the same actions as their male "
        "counterparts even though they have done nothing wrong in moral and social reality. Women may be subject to "
        "moral suspicion and consternation for violating edits of the patriarchal rule book. For example, as a female "
        "software engineer, I feel like my source code is heavily scrutinized by my male teammates. When I submit "
        "similar work as my male co-workers, I tend to receive more critiques compared to my colleagues despite our "
        "work being identical in logic and performance.",
    '3':
        "Arises due to systematic biases that afflict women software developers as a social group that has "
        "historically been and to some extent remains unjustly socially subordinate. The group members experiences "
        "challenges as being regarded as less credible when making claims about certain matters, or against certain "
        "people, hence being denied the epistemic status of knowers For example, I am a woman software developer. I "
        "find that when I present an ideas to my development team, they often ignore my input. However, when my male "
        "colleague repeats the same ideas in a follow-up meeting, the team almost immediately accepts them.",
    '4':
        "People are (often unwittingly) motivated to maintain gender hierarchies by applying social penalties to "
        "women software developers who compete for, or otherwise threaten to advance to, high-status, masculine-coded "
        "positions. This is experienced when women in such positions who are agentic are perceived as extreme in "
        "masculine-coded traits like being arrogant and aggressive. For example, as one of the female programmers in "
        "our team, I sometimes experience a sense of hostility when I provide constructive criticism or potential "
        "improvements to my male counterparts' source code. I give the same type of feedback to my female colleagues "
        "and receive praises."
}


class SklearnSentenceTransformer(BaseEstimator, TransformerMixin):
    np.random.seed(42)

    def __init__(self, model_name='WSDE_model'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        # SBERT does not require fitting in the traditional sense
        return self

    def predict(self, X):
        data_embeddings = self.model.encode(X, convert_to_tensor=True)

        definitions_embeddings = []
        for key, value in definitions.items():
            definitions_embeddings.append(self.model.encode(value, convert_to_tensor=True))

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
            closest_definitions.append(int(closest_definition))
            closest_definition_scores.append(closest_definition_score)

        return closest_definitions


    def transform(self, X):
        # Encode sentences into embeddings
        return self.model.encode(X, convert_to_tensor=True).cpu().numpy()

    def get_params(self, deep=True):
        return {"model_name": self.model_name}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = SentenceTransformer(self.model_name)
        return self
