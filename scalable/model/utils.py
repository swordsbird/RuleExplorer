from .abalone import Model as AbaloneModel
from .bankruptcy import Model as BankruptcyModel
from .cancer import Model as CancerModel
from .german_credit import Model as GermanModel
from .wine import Model as WineModel
from .stock_step0 import Model as StockModelStep0
from .stock_step1 import Model as StockModelStep1
from .stock_step2 import Model as StockModelStep2
from .stock_step3 import Model as StockModelStep3
from .stock_step4 import Model as StockModelStep4
from .credit_card import Model as CreditModel
from .credit_card_step0 import Model as CreditModel1
from .credit_card_step1 import Model as CreditModel2
from .credit_card_step2 import Model as CreditModel3
from .drybean import Model as DrybeanModel
from .obesity import Model as ObesityModel
from .crime import Model as CrimeModel
from .adult import Model as AdultModel

def get_model(data_name, model_name, tag=''):
    if data_name == 'abalone':
        return AbaloneModel(model_name)
    elif data_name == 'bankruptcy':
        return BankruptcyModel(model_name)
    elif data_name == 'stock_step0':
        return StockModelStep0(model_name)
    elif data_name == 'stock_step1':
        return StockModelStep1(model_name)
    elif data_name == 'stock_step2':
        return StockModelStep2(model_name)
    elif data_name == 'stock_step3':
        return StockModelStep3(model_name)
    elif data_name == 'stock_step4':
        return StockModelStep4(model_name)
    elif data_name == 'credit' or data_name == 'Credit Card':
        return CreditModel(model_name)
    elif data_name == 'credit1' or data_name == 'Credit Card1':
        return CreditModel1(model_name)
    elif data_name == 'credit2' or data_name == 'Credit Card2':
        return CreditModel2(model_name)
    elif data_name == 'credit3' or data_name == 'Credit Card3':
        return CreditModel3(model_name)
    elif data_name == 'cancer':
        return CancerModel(model_name)
    elif data_name == 'german' or data_name == 'German Credit':
        return GermanModel(model_name)
    elif data_name == 'drybean':
        return DrybeanModel(model_name)
    elif data_name == 'obesity':
        return ObesityModel(model_name)
    elif data_name == 'adult':
        return AdultModel(model_name)
    elif data_name == 'crime':
        return CrimeModel(model_name)
    else:#if data_name == 'wine':
        return WineModel(model_name)
