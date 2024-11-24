german_credit_encoding = {
    "credit_risk" : ["No", "Yes"], 
    "credit_history" : [
        "delay in paying off in the past",
        "critical account/other credits elsewhere",
        "no credits taken/all credits paid back duly",
        "existing credits paid back duly till now",
        "all credits at this bank paid back duly"
    ],
    "purpose" : [
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
    "installment_rate": ["< 20", "20 to 25",  "25 to 35", ">= 35"],
    "present_residence": [
        "< 1 yr", 
        "1 to 4 yrs",
        "4 to 7 yrs", 
        ">= 7 yrs"
    ],
    "number_credits": ["1", "2 to 3", "4 to 5", "6 or more"],
    "people_liable": ["0 to 2", "3 or more"],
    "savings": [
        "unknown/no savings account",
        "< 100 DM", 
        "100 to 500 DM",
        "500 to 1000 DM",
        ">= 1000 DM"
    ],
    "employment_duration": [
        "unemployed", 
        "< 1 yr", 
        "1 to 4 yrs",
        "4 to 7 yrs", 
        ">= 7 yrs"
    ],
    "personal_status_sex": [
        "not married male",
        "married male"
    ],
    "other_debtors": [
        "none",
        "co-applicant",
        "guarantor"
    ],
    "property": [
        "real estate",
        "building soc. savings agr./life insurance", 
        "car or other",
        "unknown / no property"
    ],
    "other_installment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "job": ["unemployed, unskilled and non-resident", "unskilled and resident", "skilled", "highly skilled"],
    "status": [
        "no checking account",
        "... < 0 DM",
        "0 to 200 DM",
        "... >= 200 DM (salary)"
    ],
    "telephone": ["No", "Yes"],
    "foreign_worker": ["No", "Yes"]
}

stock_encoding = {
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