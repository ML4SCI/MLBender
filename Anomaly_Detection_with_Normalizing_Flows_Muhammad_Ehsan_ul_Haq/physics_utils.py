import numpy as np
from pyjet import cluster, DTYPE_PTEPM


def convert_data_to_jets(data):
    """
    Convert raw Anomaly Detection data (events data) to jets.
    """
    leadpT = []
    alljets = []
    for i in range(data.shape[0]):
        pseudojets_input = np.zeros(len([x for x in data[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
        j_idx = 0
        for j in range(700):
            if data[i][j * 3] > 0:
                pseudojets_input[j_idx]['pT'] = data[i][j * 3]
                pseudojets_input[j_idx]['eta'] = data[i][j * 3 + 1]
                pseudojets_input[j_idx]['phi'] = data[i][j * 3 + 2]
                j_idx += 1
        sequence = cluster(pseudojets_input, R=1.0, p=-1)
        jets = sequence.inclusive_jets(ptmin=20)
        leadpT += [jets[0].pt]
        alljets += [jets]

    return leadpT, alljets


def jets_to_mjj(jets):
    """
    Convert jets to Mjj.
    """
    mjj = []
    for k in range(len(jets)):
        E = jets[k][0].e + jets[k][1].e
        px = jets[k][0].px + jets[k][1].px
        py = jets[k][0].py + jets[k][1].py
        pz = jets[k][0].pz + jets[k][1].pz
        mjj += [(E ** 2 - px ** 2 - py ** 2 - pz ** 2) ** 0.5]

    return mjj