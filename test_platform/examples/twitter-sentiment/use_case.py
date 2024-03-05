import logging
import random
import zipfile
from pathlib import Path
import argparse
from joblib import Memory
import json
import openai
import os

import numpy as np
from transformers import pipeline

import requests

import pandas as pd
from functools import cached_property

from picrystal_test import embedders, perturbers
from picrystal_test.embedders import PronounsEmbedder
import picrystal_test.core
import picrystal_test.test_catalog

labels_map = {'positive': 1, 'negative': -1, 'neutral': 0}


class SimpleEthnicityEmbedder(embedders.Embedder):
    '''This is an example of ethnicity embedder, there is no goal to be even a near production ready'''

    hispanic_or_latino = [
        "mexican", "puerto rican", "cuban", "dominican", "central american",
        "south american", "spanish", "latin", "latino", "latinx", "hispanic",
        "chican", "spanish-speaking"
    ]

    white = [
        "german", "irish", "english", "italian", "polish", "french", "scottish",
        "scandinavian", "slavic", "caucasian", "euro-american", "western", "white"
    ]

    black_or_african_american = [
        "african", "caribbean", "west indian", "somali", "nigerian", "ethiopian",
        "african american", "haitian", "black", "afro", "afro-american", "african american",
        "person of color",
    ]

    native_hawaiian_or_pacific_islander = [
        'hawaii' "native hawaiian", "samoan", "guamanian", "chamorro", "fijian", "tongan",
        "maori", "polynesian", "micronesian", "pacific islander", "polynesian", "micronesian",
        "native hawaiian"
    ]

    asian = [
        "chinese", "filipino", "asian indian", "vietnamese", "korean", "japanese", "thai", "indonesian",
        "burmese", "pakistani", "asian", "east asian", "south asian", "southeast asian"
    ]

    native_america_or_alaska_native = [
        "cherokee", "navajo", "sioux", "chippewa", "choctaw", "lumbee", "inupiat", "yupik",
        "aleut", "native american", "american indian", "first nations", "indigenous", "alaska native",
        "tribal",
    ]

    ethnicities = [
        hispanic_or_latino,
        white,
        black_or_african_american,
        native_hawaiian_or_pacific_islander,
        asian,
        native_america_or_alaska_native
    ]

    def __init__(self, tags=tuple()):
        self._tags = ('inputs', 'categorical', 'race',
                      'ethnicity', 'provides-groups-info') + tags

    @property
    def tags(self):
        return self._tags

    @property
    def groups(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    def __call__(self, inputs):
        inputs, targets, predictions = inputs
        results = np.zeros((len(inputs), ))

        for i, sentence in enumerate(inputs):
            sentence = sentence.lower()
            existed_ethnicites = []
            for j, keywords in enumerate(self.ethnicities):
                if any(word in sentence for word in keywords):
                    existed_ethnicites.append(j)

            if len(existed_ethnicites) == 0:
                # None of ethnicity mentioned
                results[i] = 6
            if len(existed_ethnicites) == 1:
                # only one ethnicity is mentioned
                results[i] = existed_ethnicites[0]
            else:
                # mentioned two or more ethnicities
                results[i] = 7
        return results

    @property
    def missing_label(self):
        return 6

    def info(self):
        return {
            'name': 'Text mentions an ethnicity',
            'tags': self.tags,
            'groups': {
                0: "Hispanic or Latino",
                1: "White",
                2: "Black or African American",
                3: "Native Hawaiian or Pacific Islander",
                4: "Asian",
                5: "Native American or Alaska Native",
                6: "None",
                7: "Two or more",
            }
        }


class IdentityEmbedderGroupInfo(embedders.IdentityEmbedder):

    def __init__(self, on, class_info=None, tags=tuple()):
        super().__init__(on, class_info=class_info, tags=tags)
        self._tags = self._tags + ('provides-groups-info',)

    @property
    def groups(self):
        return [-1, 0, 1]

    def __call__(self, inputs):
        ''' Convert hugging face result to [-1. 0. 1] '''
        inputs, targets, predictions = inputs

        score_sorted_predictions = []
        for prediction in predictions:
            score_sorted_predictions.append(
                list(sorted(prediction, key=lambda p: p['score'], reverse=True)))

        predictions = [labels_map[p[0]['label']]
                       for p in score_sorted_predictions]
        predictions = np.array(predictions)
        return super().__call__((inputs, targets, predictions))


class TransformersBinarizerEmbedder(embedders.NegPosEmbedder):

    def __call__(self, inputs):
        '''Handle hugging face result format'''
        inputs, targets, predictions = inputs

        score_sorted_predictions = []
        for prediction in predictions:
            score_sorted_predictions.append(
                list(sorted(prediction, key=lambda p: p['score'], reverse=True)))

        predictions = [labels_map[p[0]['label']]
                       for p in score_sorted_predictions]
        predictions = np.array(predictions)
        return super().__call__((inputs, targets, predictions))


class PredictionProbabilityEmbedder(embedders.Embedder):

    def __init__(self, tags=tuple()):
        self._tags = ('predictions', 'probabilites') + tags

    def __call__(self, inputs):
        '''Get probabilities in an array with shape (n, 3)
        where (:, 0) probability of negative class
              (:, 1) probability of neutral class
              (:, 2) probability of positive class
        '''
        inputs, targets, predictions = inputs

        # Sort every predictions, so outputs matrix will be always negative, neutral, positive
        label_sorted_prediction = []
        for prediction in predictions:
            prediction.sort(key=lambda p: labels_map[p['label']])
            label_sorted_prediction.append(
                list(sorted(prediction, key=lambda p: labels_map[p['label']])))

        probabilities = [[pc['score'] for pc in p]
                         for p in label_sorted_prediction]
        probabilities = np.array(probabilities)

        return probabilities

    @property
    def tags(self):
        return self._tags

    def info(self):
        return {
            'name': 'class probabilites',
            'tags': self.tags
        }


memory = Memory('./gpt_cache')  # I commit it, to share different querries


openai_api_key = os.getenv('OPENAI_TOKEN')
if openai_api_key is None:
    logging.warning(
        'No opeanai api token provided, GPT powered perturbers are off')
else:
    openai.api_key = openai_api_key


@memory.cache
def gpt_query(text, prompt):
    # Queries GPT requires to pay every time, use @memory.cache persistent cache
    # to store results
    results = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text}
        ]
    )
    return results.choices[0]['message']['content']


class GptPoweredRephrasePerturber:
    '''Use OpenAI GPT to perturbe the data by promt'''
    # TODO: Move to pertrubers file

    def __init__(self, name, prompt, n_sample=None):
        self._name = name
        self._n_sample = n_sample
        self._prompt = prompt

    def sample(self, inputs):
        # We had a big dataset, and sending all tweets to gpt is too much
        # So Perturber implements sample, function, that takes `self._n_sample` ~ 50 points out of
        # the whole dataset
        # Robustness test should checks if perturber has the `sample` function
        # And if so, it starts the graph with `Input -> Perturber.sample -> Embedder/Perturber -> Compare`
        # So instead of the whole dataset only subsample is used
        #
        # This is the first and hucky way how not to send too many points to GPT
        # without manually decreasing the dataset
        if self._n_sample is None:
            return inputs

        inputs, targets, predictions = inputs
        if len(inputs) <= self._n_sample:
            return (inputs, targets, predictions)

        np.random.seed(42)
        indices = np.random.choice(range(len(inputs)), size=self._n_sample)
        return inputs[indices], targets[indices], [predictions[i] for i in indices]

    @property
    def tags(self):
        return ('gpt',)

    def info(self):
        return {
            'name': self._name,
            'prompt': self._prompt,
            'sample': self._n_sample,
        }

    def __call__(self, inputs):
        inputs, targets, _ = inputs
        results = []
        for text in inputs:
            results.append(gpt_query(text, self._prompt))
        return np.array(results), targets


class Exampler:
    # Picks an example from the dataset by index
    #
    # One more hacky way how enable examples for tests
    # Tests should not know which exacly format of data is used
    # So UseCase.py should define how is it possible to pick an example,
    # and how to transform it to string.
    #
    # One more hacky way, that has to be rethought
    def __call__(self, inputs, index):
        inputs, targets, predictions = inputs
        prediction = predictions[index]
        target = targets[index]
        predicted_label = labels_map[max(
            prediction, key=lambda x: x['score'])['label']]

        example = inputs[index]
        return {
            'data': example,
            'format': 'text',
            'predicted_label': int(predicted_label),
            'true_label': int(target)
        }


class TwitterSentimentUseCase:

    def __init__(self, n_points, dev=False):
        self._n_points = n_points
        self._dev = dev

    def setup(self, force=False):
        url = 'https://storage.googleapis.com/picrystal-bucket/twitter-sentiment/3c02f9d3-ed35-49aa-bc90-8dcf782f4b03_Tweets.csv'
        self.data_all = pd.read_csv(url)
        self.data_all = self.data_all.dropna()

        eth_emb = SimpleEthnicityEmbedder()
        gender_emb = PronounsEmbedder()

        # There is too few points in the whole dataset that has words
        # required by ethnicity and gender embedders
        # So we manually select only points that have this words. So our dataset
        # looks more balanced
        self.data_all['ethnicity'] = eth_emb(
            (self.data_all['text'], None, None))
        self.data_all['gender'] = gender_emb(
            (self.data_all['text'], None, None))

        self.eth_data = self.data_all.loc[self.data_all['ethnicity'].isin(
            [0, 1, 2, 3, 4, 5, 6])]
        self.gender_data = self.data_all.groupby(by=['gender']).apply(
            lambda x: x.sample(n=min(len(x), 100), random_state=42)
        )
        self.data = pd.concat([self.eth_data, self.gender_data], axis=0)
        self.data = self.data.drop_duplicates()

        if self._n_points:
            self.data = self.data.sample(
                n=min(self.data.shape[0], self._n_points), random_state=42)

        self.data_x = self.data['text']
        self.data_y = self.data['sentiment']
        self.data_y = self.data_y.replace(labels_map)

    @cached_property
    def model(self):
        model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        pipe = pipeline("sentiment-analysis", model=model_path,
                        framework="pt", top_k=3)

        def sentiment_model(data_x):
            if isinstance(data_x, np.ndarray):
                data_x = data_x.tolist()
            preds = pipe(data_x)
            return preds
            # return np.array([p['score'] for p in preds])  # for probabilities

        return sentiment_model

    @cached_property
    def inputs(self):
        return self.data_x.values

    @cached_property
    def targets(self):
        return self.data_y.values

    class_info = [
        {'value': -1, 'name': 'negative'},
        {'value': 0, 'name': 'neutral'},
        {'value': 1, 'name': 'positive'}
    ]

    embedders = [
        IdentityEmbedderGroupInfo(
            on='groundtruth', class_info=class_info, tags=('categorical', 'robustness')),
        IdentityEmbedderGroupInfo(
            on='predictions', class_info=class_info, tags=('categorical', 'robustness')),
        PredictionProbabilityEmbedder(),
        TransformersBinarizerEmbedder(
            on='groundtruth', positive=[-1], class_info=class_info, meaning='negative-vs-all'),
        TransformersBinarizerEmbedder(
            on='predictions', positive=[-1], class_info=class_info, meaning='negative-vs-all'),
        TransformersBinarizerEmbedder(on='groundtruth', positive=[
                                      1], class_info=class_info, meaning='positive-vs-all'),
        TransformersBinarizerEmbedder(on='predictions', positive=[
                                      1], class_info=class_info, meaning='positive-vs-all'),
        TransformersBinarizerEmbedder(on='groundtruth', positive=[
                                      0], class_info=class_info, meaning='neutral-vs-all'),
        TransformersBinarizerEmbedder(on='predictions', positive=[
                                      0], class_info=class_info, meaning='neutral-vs-all'),
        PronounsEmbedder(tags=('sensitive', 'gender', 'sex')),
        SimpleEthnicityEmbedder(
            tags=('sensitive', 'race', 'nationality', 'ethnicity')),
    ]

    @property
    def perturbers(self):
        dev_perturbers = [
            perturbers.RandomIntSuffixPerturber(),
            perturbers.TypoPerturber(aug_char_max=1, aug_word_p=0.1),
            perturbers.TypoPerturber(aug_char_max=5, aug_word_p=0.5),
        ]

        more_perturbers = [
            perturbers.TypoPerturber(aug_char_max=1, aug_word_p=0.2),
            perturbers.TypoPerturber(aug_char_max=1, aug_word_p=0.5),

            perturbers.TypoPerturber(aug_char_max=3, aug_word_p=0.1),
            perturbers.TypoPerturber(aug_char_max=3, aug_word_p=0.3),
            perturbers.TypoPerturber(aug_char_max=3, aug_word_p=0.5),

            perturbers.TypoPerturber(aug_char_max=5, aug_word_p=0.1),
            perturbers.TypoPerturber(aug_char_max=5, aug_word_p=0.3),
        ]

        if openai_api_key is not None:
            n_samples = 100 if not self._dev else 10
            gpt_perturbers = [
                GptPoweredRephrasePerturber(n_sample=n_samples, name='Rephrase',
                                            prompt='You are rephrase bot, simply rephrase every tweet that the user provides to you, without adding or deleting information.'),
                GptPoweredRephrasePerturber(n_sample=n_samples, name='Rephrase by adding adjectives',
                                            prompt='You are rephrase bot, rephrase every tweet that the user provides to you so that the rephrased tweets contain a lot of adjectives, however do not add or delete information.'),
                GptPoweredRephrasePerturber(n_sample=n_samples, name='Rephrase and introduce grammar mistakes',
                                            prompt='You are rephrase bot, rephrase every tweet that the user provides to you with more grammatical mistakes, however do not add or delete information.'),
            ]
        else:
            gpt_perturbers = []
        if self._dev:
            return dev_perturbers + gpt_perturbers
        else:
            return dev_perturbers + more_perturbers + gpt_perturbers

    exampler = Exampler()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trust-profile', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-points', default=None, type=int,
                        help='Sample out of all dataset n points. When no present the whole dataset is used')
    parser.add_argument('--dev', action='store_true',
                        help='When present, smaller parameters for test will be used')

    args = parser.parse_args()

    usecase = TwitterSentimentUseCase(args.n_points, args.dev)
    usecase.setup(force=True)

    with open(args.trust_profile) as f:
        trust_profile = json.load(f)

    result = picrystal_test.core.run_all_tests(
        trust_profile,
        usecase,
        picrystal_test.test_catalog.catalog
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)
