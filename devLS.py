import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from adftPerformance.models.MJ1 import MJ1_Validator, MJ1_Preprocessor, MJ1_Predictor

def loaddata(path):
    dataframes = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            dataframes.append(pd.read_json(os.path.join(path, file)))

    transp = []
    for df in dataframes:
        transp.append(df.transpose())
    
    return pd.concat(transp)

def confPredictor(path):

    val = MJ1_Validator(input='smiles')
    prep = MJ1_Preprocessor(optimize=True, gradients=1600)
    pred = MJ1_Predictor(
        model_path=path,
        validator=val,
        preprocessor=prep)
    return pred

def start(predictor, param):
    with open('/home/michiel/alchemical-DFT-performance/dienes.csv', 'r') as f:
        f.readline()
        content = [line.rstrip() for line in f]
        
        with open('predictions.csv', 'w') as p:
            p.write(f'cid,smiles,{ param }\n')

            for l in tqdm(content):
                l = l.split(sep=',')
                p.write(f'{l[0]},{l[1]},{ predictor.predict(l[1]) },\n')


def add(path, predictor, param):
    with open(path, 'r+') as f:
        header = f.readline()
        content = [line.rstrip() for line in f]

        f.seek(0)

        f.write(f'{ header.rstrip() },{ param }\n')

        for l in tqdm(content):
            s = l.split(sep=',')
            f.write(f'{ l.rstrip() }{ predictor.predict(s[1]) },\n')

def predict_xyz_start(predictor, param):
    data = loaddata('/home/michiel/projects/CNOS_Dataset/datasets/OOS/global_descriptors')
    cids = data['cid'].to_numpy()
    geo = data['optimized_geometry'].to_numpy()

    with open('predictions.csv', 'w') as p:
        p.write(f'CID,prediction_{ param }\n')

        for id, m in tqdm(zip(cids, geo)):
            atoms = m[0]
            xyz = m[1]

            string_data = f'{ len(atoms) }\n\n'
            for a, coord in zip(atoms, xyz):
                temp_string = f'{ a }\t{ coord[0] }\t{ coord[1] }\t{ coord[2] }\n'
                string_data += temp_string
            
            p.write(f'{ id },{ predictor.predict(string_data) },\n')

def predict_xyz_add(path, predictor, param):
    data = loaddata('/home/michiel/projects/CNOS_Dataset/datasets/OOS/global_descriptors')
    cids = data['cid'].to_numpy()
    geo = data['optimized_geometry'].to_numpy()

    with open(path, 'r+') as f:
        header = f.readline()
        content = [line.rstrip() for line in f]

        f.seek(0)

        f.write(f'{ header.rstrip() },predicions_{ param }\n')

        for id, m, l in tqdm(zip(cids, geo, content)):
            atoms = m[0]
            xyz = m[1]

            string_data = f'{ len(atoms) }\n\n'
            for a, coord in zip(atoms, xyz):
                temp_string = f'{ a }\t{ coord[0] }\t{ coord[1] }\t{ coord[2] }\n'
                string_data += temp_string
            
            f.write(f'{ l.rstrip() }{ predictor.predict(string_data) },\n')
                
if __name__ == '__main__':

    lumo = confPredictor('/home/michiel/alchemical-DFT-performance/model_files/lumo')
    start(lumo, 'predicted_lumo_ev')
    del lumo

    homo = confPredictor('/home/michiel/alchemical-DFT-performance/model_files/homo')
    add('/home/michiel/alchemical-DFT-performance/predictions.csv', homo, 'predicted_homo_ev')
    del homo

    electro = confPredictor('/home/michiel/alchemical-DFT-performance/model_files/electro')
    add('/home/michiel/alchemical-DFT-performance/predictions.csv', electro, 'predicted_electrophilicity_index_ev')
    del electro

    #chem_pot = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_pot/chem_pot')
    #add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_pot, 'chemical_potential')
    #predict_xyz_add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_pot, 'chemical_potential')
    #del chem_pot

        #chem_hard = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_hard/chem_hard')
    #add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_hard, 'chemical_hardness')
    #predict_xyz_add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_hard, 'chemical_hardness')
    #del chem_hard