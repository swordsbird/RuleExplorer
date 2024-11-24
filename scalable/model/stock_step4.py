from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append('.')
import pandas as pd
from scalable.model.base_model import BaseModel
from scalable.model.data_encoding import stock_encoding
import numpy as np
from scalable.config import data_path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score

new_features = ['CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLHIGHWAVE', 'ATR', 'AD', 'alpha011', 'alpha029', 'alpha030', 'alpha031', 'alpha036']
current_features = ['label', 'rating', 'grossProfit', 'totalRevenue', 'costOfRevenue', 'operatingIncome', 'operatingExpenses', 'interestIncome', 'interestExpense', 'otherNonOperatingIncome', 'depreciationAndAmortization', 'incomeTaxExpense', 'interestAndDebtExpense', 'comprehensiveIncomeNetOfTax', 'ebit', 'ebitda', 'netIncome', 'totalAssets', 'totalCurrentAssets', 'inventory', 'totalNonCurrentAssets', 'goodwill', 'investments', 'totalLiabilities', 'totalCurrentLiabilities', 'payable', 'currentDebt', 'shortTermDebt', 'longTermDebt', 'totalEquity', 'retainedEarnings', 'commonStock', 'operatingCashflow', 'changeInOperatingLiabilities', 'changeInOperatingAssets', 'changeInReceivables', 'profitLoss', 'cashflowFromInvestment', 'proceedsFromRepurchase', 'changeInCash', 'reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage', 'currentRatio', 'totalCashflowTTM', 'netIncomeTTM', 'totalRevenueTTM', 'totalLiabilitiesTTM', 'ebitTTM', 'ebitdaTTM', 'retainedEarningsTTM', 'totalEquityTTM', 'totalShares', 'netIncomeParent', 'netIncomeParentTTM', 'KMID', 'KLEN', 'KMID2', 'KUP2', 'KLOW', 'KSFT2', 'OPEN0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'MA20', 'MA30', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MIN5', 'MIN60', 'QTLU10', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP60', 'CNTD60', 'SUMN5', 'SUMD5', 'VMA10', 'VMA30', 'VMA60', 'VSTD20', 'VSTD60', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'marketCap', 'peRatioTTM', 'pcfRatioTTM', 'pbRatioTTM', 'psRatioTTM', 'evToEbit', 'earningsGrowth', 'revenuePerShareTTM', 'returnOnEquityTTM', 'returnOnAsset', 'returnOnAssetTTM', 'returnOnAssetNetTTM', 'netIncomeToRevenue', 'netIncomeToRevenueTTM', 'ebitToRevenue', 'ebitToRevenueTTM', 'cashflowPerShareTTM', 'cashflowRatio', 'retainedEarningPerShare', 'retainedEarningPerShareTTM', 'debtToAssetRatio', 'currentAssetRatio', 'revenueGrowth', 'country_United States', 'country_China', 'country_Others', 'country_Israel', 'country_Canada', 'country_United Kingdom', 'exchange_NMS', 'exchange_NCM', 'exchange_NGM', 'exchange_ASE', 'exchange_NYQ', 'exchange_Others', 'NATR']
current_features = current_features + new_features
remove_features =  ['VMA20', 'CORD60', 'MIN5', 'CORD5', 'MIN60', 'ROC10', 'KLEN', 'netIncome', 'ATR']
current_features = [k for k in current_features if k not in remove_features]

random_state = 42

class Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.data_name = 'stock'
        self.data_path = os.path.join(data_path, 'case2_stock/step/3year_5.csv')
        self.data_table = pd.read_csv(self.data_path)

        self.test_data_path = os.path.join(data_path, 'case2_stock/step/3month_5.csv')
        self.test_data_table = pd.read_csv(self.test_data_path)

        self.target = 'label'
        self.output_labels = ["decrease", "increase", "stable"]
        self.model_id = 111

        self.has_categorical_feature = True
        self.model_name = model_name
        if model_name == 'rf' or model_name == 'random forest':
            self.parameters = {
                'n_estimators': 150,
                'max_depth': 30,
                'random_state': random_state,
            }
        else:
            self.parameters = {
                'n_estimators': 500,
                'learning_rate': 0.0066959005792512315,
                'colsample_bytree': 0.901839117838658,
                'subsample': 0.9225884665248404,
                'num_leaves': 85,
                'lambda_l1': 4.015012029029533,
                'lambda_l2': 3.9572661746980793,
                'class_weight': 'balanced',
                'random_state': random_state,
                'verbosity': -1,
            }

    def init_data(self):
        self.data_table = self.data_table.drop('date', axis=1)
        self.test_data_table = self.test_data_table.drop('date', axis=1)

        data_table = self.data_table.drop('ticker', axis=1)
        data_table = data_table.drop('newPrice', axis = 1)
        data_table = data_table.drop('currentPrice', axis = 1)

        # data_table['peRatioTTM'] = 1.0 / data_table['peRatioTTM']
        # data_table['evToEbit'] = 1.0 / data_table['evToEbit']
        for k in data_table.columns:
            if 'industry' in k or 'sector' in k:
                data_table = data_table.drop(k, axis = 1)
        data_table = data_table[current_features]

        features = data_table.columns.tolist()
        features = [k for k in features if k != 'rating' and k != 'label']
        print(f'{len(features)} features')

        for key in stock_encoding:
            index = 0
            for i in range(len(features)):
                if key in features[i]:
                    features[i] = key + '_' + stock_encoding[key][index]
                    index += 1

        X_train = data_table[features]
        y_train = data_table[self.target]
        X_test = self.test_data_table[features]
        y_test = self.test_data_table[self.target]

        self.train_rating = data_table['rating'].values
        self.test_rating = self.test_data_table['rating'].values
        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_test = y_test.values
        self.X = X_train.values
        self.y = y_train.values
        self.data_table = data_table.drop('rating', axis = 1)
        self.current_features = features

        self.check_columns(self.data_table, self.target)

if __name__ == '__main__':
    model = Model('lightgbm')
    model.init_data()
    # print(model.X_train.mean())
    model.train()
    model.get_performance()

    y_pred = model.clf.predict(model.X_train)
    conf_mat = confusion_matrix(model.y_train, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Train Accuracy on {model.output_labels[i]}: {accuracy}')

    y_pred = model.clf.predict(model.X_test)
    conf_mat = confusion_matrix(model.y_test, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Test Accuracy on {model.output_labels[i]}: {accuracy}')

    bank_idx = np.flatnonzero(model.test_data_table['industry_Banks—Regional'])
    X_test = model.X_test[bank_idx]
    y_test = model.y_test[bank_idx]
    y_pred = model.clf.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracys = []
    model.output_labels = model.clf.classes_
    num_classes = len(model.output_labels)
    for i in range(num_classes):
        accuracy = conf_mat[i, i] / conf_mat[i].sum()
        accuracys.append(accuracy)
        print(f'Bank Test Accuracy on {model.output_labels[i]}: {accuracy}')
    accuracy = accuracy_score(y_test, y_pred)
    print(round(accuracy, 4))
    print(conf_mat)

    clf = model.clf
    sorted_features = [(clf.feature_name_[i], clf.feature_importances_[i]) for i in range(len(clf.feature_importances_))]
    sorted_features = sorted(sorted_features, key = lambda x: -x[1])
    for k in sorted_features[:20]:
        i, j = k
        i = model.current_features[int(i.split('_')[1])]
        print(i, j)


    real_data_path = os.path.join(data_path, 'case2_stock/step/real_world.csv')
    test_df = pd.read_csv(real_data_path)
    features = ['ticker'] + current_features
    selected_features = features[3:]
    start = 1
    for week, d in enumerate(test_df['date'].unique()):
        df1w = test_df[test_df['date'] == d]
        df1w = df1w[features]
        X_test = df1w[selected_features]
        y_test = df1w['label']

        y_pred_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        idx = np.zeros(len(X_test)) > 0
        ratios = []
        stocks = []
        mcap = []
        for i in np.argsort(y_pred_prob[:, 1])[::-1][:20]:
            cap = df1w['marketCap'].values[i]
            if cap > 2e8:
                ratios.append(df1w['rating'].values[i])
                stocks.append(df1w['ticker'].values[i])
                mcap.append(df1w['marketCap'].values[i])
        ratio = np.mean(ratios)
        start = start * ratio
        conf_mat = confusion_matrix(y_test, y_pred)
        accuracys = []

        tot = 0
        print(f'week{week}', start, ratio)
        for i in range(clf.n_classes_):
            accuracy = conf_mat[i, i] / conf_mat[i].sum()
            accuracys.append(accuracy)
            print(f'Accuracy on {clf.classes_[i]}: {accuracy}')
            tot += conf_mat[i, i]
        tot /= len(y_pred)
        print(f'Accuracy: {tot}')
        #for i in np.argsort(ratios)[::-1][:5]:
        #    print(stocks[i], mcap[i], ratios[i])
        print()

    model.generate_path()

