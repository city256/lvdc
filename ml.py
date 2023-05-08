import db

def predict_load():
    pv_df = db.get_pv_monitor(db.conn)
    print(pv_df)
    return

def predict_pv():

    return


def calulate_pf(limit=None, time=None):

    if (limit!=None):
        predict_load()

    pf = 12.1

    return pf