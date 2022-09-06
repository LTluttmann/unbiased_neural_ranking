import numpy as np
from mt.archive.prepare_data_for_ranknet import exchange_rows

def test_exchange_rows():
    a = np.array([[20, 1],
            [25,1],
            [30,1],
            [35,1]])

    b = np.array([[7,0],
                 [10, 0],
                 [12, 0],
                 [5, 0]])
    
    c, d = exchange_rows(a,b)
    
    assert c[:,-1].sum() == 2
    assert d[:,-1].sum() == 2
    
    # iterate through all rows of the input arrays
    for row in np.concatenate((a,b),axis=0):
        assert row.tolist() in c.tolist() or row.tolist() in d.tolist()