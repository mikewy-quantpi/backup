import argparse
import json
from functools import cached_property
from pathlib import Path
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from picrystal_test import embedders
from picrystal_test import perturbers

import picrystal_test.core
import picrystal_test.test_catalog


class HiringUseCase:
    target = 'HiredOrNot'
    predictors = [
        'State',
        'Sex',
        'MaritalDesc',
        'CitizenDesc',
        'RaceDesc',
        'Department',
        'RecruitmentSource',
        'PerformanceScore',
        'SpecialProjectsCount'
    ]

    def __init__(self):
        self.train_df, self.val_df = train_test_split(
            self.df, test_size=0.2, random_state=2018, shuffle=True)

        self.model  # train the model

    @cached_property
    def df(self):
        df = pd.read_csv(
            'https://storage.googleapis.com/picrystal-bucket/hiring/445b7773-3431-4c34-a762-ce8986670aa3_main_hiring_updated.csv')
        df = df.drop('Unnamed: 0', axis=1)

        return df

    @cached_property
    def model(self):
        clf = LogisticRegression(random_state=42, penalty='l2')
        clf.fit(self.train_df[self.predictors].values,
                self.train_df[self.target].values)
        return clf.predict

    @cached_property
    def inputs(self):
        return self.val_df[self.predictors].values

    @cached_property
    def targets(self):
        return self.val_df[self.target].values

    embedders = [
        embedders.IdentityEmbedder(on='predictions', tags=('probabilities',)),

        embedders.BinaryEmbedder(on='groundtruth'),
        embedders.BinaryEmbedderFromProbability(
            on='predictions', threshold=0.5),

        embedders.CategoricalIdentityInputEmbedder(
            column=1,
            groups={1: 'male', 2: 'female'},
            tags=('sensitive', 'gender', 'sex'),
            name='gender'
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=2,
            groups={1: 'married', 2: 'single',
                    3: 'divorced', 4: 'widowed', 5: 'separated'},
            tags=('marriage',),
            name='marriage'
        ),
        embedders.CategoricalIdentityInputEmbedder(
            column=3,
            groups={1: 'US Citizen', 2: 'Eligible NonCitizen', 3: 'Non-Citizen'},
            tags=('sensitive', 'citizenship',),
            name='citizenship'
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=4,
            groups={1: 'white', 2: 'black or african american', 3: 'two or more races',
                    4: 'asian', 5: 'american indian or alaska native', 6: 'hispanic'},
            tags=('sensitive', 'race', ),
            name='race',
            missing_label=3,
        ),

    ]

    perturbers = [
        perturbers.RandomShufflePerturber(
            column_indices_to_shuffle=[0], column_name='state'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          1], column_name='gender'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          2], column_name='marirtal description'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          3], column_name='citizen description'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          4], column_name='race description'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          5], column_name='department'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          6], column_name='recruitment source'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          7], column_name='performance score'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          8], column_name='special projects count'),
    ]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trust-profile', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    usecase = HiringUseCase()

    with open(args.trust_profile) as f:
        trust_profile = json.load(f)

    result = picrystal_test.core.run_all_tests(
        trust_profile,
        usecase,
        picrystal_test.test_catalog.catalog
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
