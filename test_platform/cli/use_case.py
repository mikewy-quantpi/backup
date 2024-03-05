import argparse
import json
from functools import cached_property
from pathlib import Path
import requests
import pandas as pd

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from picrystal_test import embedders
import picrystal_test.core
import picrystal_test.test_catalog
from copy import deepcopy

class AgePerturber:

    @property
    def tags(self):
        return ('age',)

    def info(self):
        return {'name': 'age perturber'}

    def __call__(self, inputs):
        inputs, targets, _ = deepcopy(inputs)

        inputs[:, 4] = inputs[:, 4] + 10

        return (inputs, targets)




class CreditCardUseCase:

    target = 'default.payment.next.month'
    predictors = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                  'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    def __init__(self):
        self.setup(force=False)
        self.train_df, self.val_df = train_test_split(self.df, test_size=0.2, random_state=2018, shuffle=True)
        self.model  # train the model

    def setup(self, force=False):
        folder = Path.home() / Path('.piclient')
        if not folder.exists():
            folder.mkdir()
        elif not folder.is_dir():
            raise RuntimeError(f'{folder} is not a directory')

        credit_card_folder = folder / 'credit-default'
        credit_card_folder.mkdir(exist_ok=True)

        file = credit_card_folder / 'UCI_Credit_Card.csv'

        if force or not file.exists():
            url = 'https://storage.gra.cloud.ovh.net/v1/AUTH_0ee95427106749e7b31e40e2e2351d54/mlresources/credit-default/UCI_Credit_Card.csv'

            response = requests.get(url, stream=False)

            response.raise_for_status()

            with open(file, "wb") as handle:
                handle.write(response.content)

    @cached_property
    def df(self):
        f = Path.home() / Path('.piclient') / 'credit-default' / 'UCI_Credit_Card.csv'
        return pd.read_csv(f)

    @cached_property
    def model(self):
        lgb_train = lgb.Dataset(self.train_df[self.predictors], self.train_df[self.target])
        lgb_eval = lgb.Dataset(self.val_df[self.predictors], self.val_df[self.target])

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
        embedders.BinaryEmbedderFromProbability(on='predictions', threshold=0.5),

        # 1 = male; 2 = female
        embedders.PrivilegedEmbedder(column=1, unprivileged=[2], tags=('sensitive', 'gender'), name='gender'),

        # 1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
        embedders.PrivilegedEmbedder(column=2, unprivileged=[0, 1, 4, 5, 6], tags=('education', ), name='education'),
        
        # Marital status (1=married, 2=single, 3=others)
        embedders.PrivilegedEmbedder(column=3, unprivileged=[0, 1, 3], tags=('marriage', ), name='marriage'),

        embedders.PrivilegedEmbedder(column=4, unprivileged=range(45, 100), tags=('age', 'sensitive'), name='age'),
    ]

    perturbers = [AgePerturber()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trust-profile', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    usecase = CreditCardUseCase()
    usecase.setup(force=True)

    with open(args.trust_profile) as f:
        trust_profile = json.load(f)

    result = picrystal_test.core.run_all_tests(
        trust_profile,
        usecase,
        picrystal_test.test_catalog.catalog
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
