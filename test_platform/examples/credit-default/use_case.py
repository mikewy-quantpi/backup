import argparse
import json
from functools import cached_property
from pathlib import Path
import requests
import pandas as pd

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from picrystal_test import embedders
from picrystal_test import perturbers

import picrystal_test.core
import picrystal_test.test_catalog


class AgePerturber:

    @property
    def tags(self):
        return ('age',)

    def info(self):
        return {'name': 'age perturber'}

    def __call__(self, inputs):
        inputs, targets, _ = inputs

        inputs[:, 4] = inputs[:, 4] + 10

        return (inputs, targets)


class CreditCardUseCase:

    target = 'default.payment.next.month'
    predictors = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                  'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    def __init__(self):
        self.train_df, self.val_df = train_test_split(
            self.df, test_size=0.2, random_state=2018, shuffle=True)
        self.model  # train the model

    @cached_property
    def df(self):
        url = 'https://storage.googleapis.com/picrystal-bucket/credit-default/5f81d46e-71fd-4db6-8d84-eb3d95296b1d_UCI_Credit_Card.csv'
        return pd.read_csv(url)

    @cached_property
    def model(self):
        lgb_train = lgb.Dataset(
            self.train_df[self.predictors], self.train_df[self.target])
        lgb_eval = lgb.Dataset(
            self.val_df[self.predictors], self.val_df[self.target])

        params = {
            'objective': 'binary',
            'metric': {'auc', 'binary_logloss'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': 0,
            'seed': 10
        }

        print('Starting training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        callbacks=[lgb.early_stopping(stopping_rounds=5)])
        return gbm.predict

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
            column=1, groups={1: 'male', 2: 'female'}, tags=('sensitive', 'gender', 'sex'), name='gender'
        ),

        embedders.CategoricalIdentityInputEmbedder(
            column=2,
            groups={1: 'graduate school', 2: 'university',
                    3: 'high school', 4: 'others', 5: 'unknown', 6: 'unkown'},
            missing_label=5,
            tags=('education', ),
            name='education'
        ),

        # Marital status (1=married, 2=single, 3=others)
        embedders.CategoricalIdentityInputEmbedder(
            column=3, groups={1: 'married', 2: 'single', 3: 'others'}, tags=('marriage', ), name='marriage',
            missing_label=3,
        ),

        embedders.GroupingInputEmbedder(
            column=4,
            groups={
                'young': range(0, 25),
                'adult': range(25, 50),
                'old': range(50, 100)
            },
            name='age',
            tags=('age', 'sensitive',)
        )
    ]

    perturbers = [
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          0], column_name='limit balance'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          1], column_name='gender'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          2], column_name='education'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          3], column_name='marriage'),
        perturbers.RandomShufflePerturber(
            column_indices_to_shuffle=[4], column_name='age'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          5], column_name='payment 1 month ago'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          6], column_name='payment 2 month ago'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          7], column_name='payment 3 month ago'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          8], column_name='payment 4 month ago'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          9], column_name='payment 5 month ago'),
        perturbers.RandomShufflePerturber(column_indices_to_shuffle=[
                                          10], column_name='payment 6 month ago'),
    ]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trust-profile', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    usecase = CreditCardUseCase()

    with open(args.trust_profile) as f:
        trust_profile = json.load(f)

    result = picrystal_test.core.run_all_tests(
        trust_profile,
        usecase,
        picrystal_test.test_catalog.catalog
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
