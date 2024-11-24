
cache_dir_path = './cache'

data_encoding = {}
data_setting = {}

# Gender,Age,Debt,Married,BankCustomer,Industry,Ethnicity,YearsEmployed,PriorDefault,Employed,CreditScore,DriversLicense,Citizen,ZipCode,Income,Approved


data_setting['stock'] = {
    'psRatioTTM': {'scale': 'linear'},
}

data_encoding['stock'] = {
    'industry': ['Biotechnology',
                 'Oil & Gas E&P',
                 'Software—Application',
                 'Semiconductors',
                 'Medical Devices',
                 'Asset Management',
                 'Software—Infrastructure',
                 'Specialty Industrial Machinery',
                 'Banks—Regional', 'Others'],
    'country': ['United States', 'China', 'Israel', 'Canada', 'United Kingdom', 'Others'],
    'exchange': ['NMS', 'NCM', 'NGM', 'ASE', 'NYQ', 'Others'],
    'sector': ['Consumer Cyclical',
               'Healthcare',
               'Technology',
               'Energy',
               'Consumer Defensive',
               'Industrials',
               'Communication Services',
               'Financial Services',
               'Basic Materials',
               'Real Estate',
               'Utilities'],
    'previousConsensus': ['neutral', 'buy', 'sell', 'strongbuy', 'strongsell']
}

data_setting['credit'] = {
    'YearsEmployed': {'scale': 'log'},
    'CreditScore': {'scale': 'log'},
    'Income': {'scale': 'log'},
    'Industry': {'display_name': 'Job'},
}

data_setting['credit1'] = data_setting['credit']
data_setting['credit2'] = data_setting['credit']
data_setting['credit3'] = data_setting['credit']

data_encoding['credit'] = {
    'Gender': ['Female', 'Male'],
    'Married': ['Single', 'Married'],
    'BankCustomer': ['No bank account', 'Have a bank account'],
    'Job': [],
    'Ethnicity': [],
    'PriorDefault': ['No prior defaults', 'Prior defaults'],
    'Employed': ['No', 'Yes'],
    'DriversLicense': ['No', 'Yes'],
    'Citizen': [],
    'ZipCode': [],
    'Approved': ['Rejected', 'Approved'],
}

data_encoding['credit4'] = data_encoding['credit']
data_encoding['credit3'] = data_encoding['credit']
data_encoding['credit_new'] = data_encoding['credit']

data_encoding['german'] = {
    'credit_risk': ['No', 'Yes'],
    'credit_history': [
        "delay in paying off in the past",
        "critical account/other credits elsewhere",
        "no credits taken/all credits paid back duly",
        "existing credits paid back duly till now",
        "all credits at this bank paid back duly",
    ],
    'purpose': [
        "others",
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business"
    ],
    'installment_rate': ["< 20", "20 <= ... < 25",  "25 <= ... < 35", ">= 35"],
    'present_residence': [
        "< 1 yr",
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs",
        ">= 7 yrs"
    ],
    'number_credits': ["1", "2-3", "4-5", ">= 6"],
    'people_liable': ["0 to 2", "3 or more"],
    'savings': [
        "unknown/no savings account",
        "... <  100 DM",
        "100 <= ... <  500 DM",
        "500 <= ... < 1000 DM",
        "... >= 1000 DM",
    ],
    'employment_duration': [
        "unemployed",
        "< 1 yr",
        "1 <= ... < 4 yrs",
        "4 <= ... < 7 yrs",
        ">= 7 yrs"
    ],
    'personal_status_sex': [
        "not married male",
        "married male",
    ],
    'other_debtors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'property': [
        "real estate",
        "building soc. savings agr./life insurance",
        "car or other",
        "unknown / no property",
    ],
    'other_installment_plans': ['bank', 'stores', 'none'],
    'housing': ["rent", "own", "for free"],
    'job': [
        'unemployed/ unskilled - non-resident',
        'unskilled - resident',
        'skilled employee / official',
        'management/ self-employed/ highly qualified employee/ officer'
    ],
    'status': [
        "no checking account",
        "... < 0 DM",
        "0<= ... < 200 DM",
        "... >= 200 DM / salary for at least 1 year",
    ],
    'telephone': ['No', 'Yes'],
    'foreign_worker': ['No', 'Yes'],
}
