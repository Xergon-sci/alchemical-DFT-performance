import os
from tqdm import tqdm
import numpy as np
import tensorflow as tfB

def confPredictor(path):

    val = MJ1_Validator(input='smiles')
    prep = MJ1_Preprocessor(optimize=True, gradients=50)

    pred = MJ1_Predictor(
        model_path=path,
        validator=val,
        preprocessor=prep)
    return pred

def start(predictor, param):
    with open('/home/michiel/projects/CNOS_Dataset/datasets/OOS_RAW/OOS_RAW_tot.csv', 'r') as f:
        f.readline()
        content = [line.rstrip() for line in f]
        
        with open('predictions.csv', 'w') as p:
            p.write(f'CID,smiles,prediction_{ param }\n')

            for l in tqdm(content):
                l = l.split(sep=',')
                p.write(f'{ l[1][:-2] },{ l[2] },{ predictor.predict(l[2]) },\n')

def add(path, predictor, param):
    with open(path, 'r+') as f:
        header = f.readline()
        content = [line.rstrip() for line in f]

        f.seek(0)

        f.write(f'{ header.rstrip() },predicions_{ param }\n')
        for l in tqdm(content):
            s = l.split(sep=',')
            f.write(f'{ l.rstrip() }{ predictor.predict(s[1]) },\n')



if __name__ == '__main__':

    lumo = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/lumo/lumo')
    start(lumo, 'LUMO')
    del lumo

    chem_pot = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_pot/chem_pot')
    add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_pot, 'chemical_potential')
    del chem_pot

    electro = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/electro/electro')
    add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', electro, 'electrophilicity_index')
    del electro

    chem_hard = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_hard/chem_hard')
    add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_hard, 'chemical_hardness')
    del chem_hard