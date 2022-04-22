class WrongEnabledBranches(Exception):
    """Something wrong with enabled branches. Check enable_sr and enable_rec in config"""
    pass

class WrongMetrucForSaveBestRec(Exception):
    """Something wrong with rec_best_model_save. Check rec_best_model_save in config. It could be 'lev_dis' or 'acc'"""
    pass

class WrongModelForSaveBestRec(Exception):
    """Something wrong with acc_best_model. Check acc_best_model in config. It could be 'ctc' or 'crnn'"""
    pass