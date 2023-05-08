import db


def predict_load():
    pv_df = db.get_pv_monitor()
    print(pv_df)
    return

def predict_pv():

    return

def data_preprocessing():
    return

def calculate_pf(limit=None, time=None, pf=None):

    if (limit!=None):
        predict_load()
    elif(time!=None):
        print(time)
    elif(pf!=None):
        print(pf)
    else:
        print()

    pf = 12.1

    return pf

def optimize_mode():
    print("optimize_mode")
    pf = calculate_pf()
    return pf

def peak_mode(limit):
    print(f'peak_mode, limit= {limit}')
    pf = calculate_pf(limit)
    return pf

def demand_mode():
    print("demand_mode")
    pf = calculate_pf()
    return pf

def pv_mode():
    print("pv_mode")
    pf = calculate_pf()
    return pf
