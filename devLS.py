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

    val = MJ1_Validator(input='xyz')
    prep = MJ1_Preprocessor(optimize=False, gradients=1600)
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

    #lumo = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/lumo/lumo')
    #start(lumo, 'LUMO')
    #predict_xyz_start(lumo, 'lumo')
    #del lumo


    #chem_pot = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_pot/chem_pot')
    #add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_pot, 'chemical_potential')
    #predict_xyz_add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_pot, 'chemical_potential')
    #del chem_pot

    #electro = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/electro/electro')
    #add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', electro, 'electrophilicity_index')
    #predict_xyz_add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', electro, 'electrophilicity_index')
    #del electro

    #chem_hard = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_hard/chem_hard')
    #add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_hard, 'chemical_hardness')
    #predict_xyz_add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', chem_hard, 'chemical_hardness')
    #del chem_hard

    homo = confPredictor('/home/michiel/projects/alchemical-DFT-performance/tests/models/homo/homo')
    #add('/home/michiel/projects/alchemical-DFT-performance/predictions_1600.csv', homo, 'homo')
    predict_xyz_add('/home/michiel/projects/alchemical-DFT-performance/predictions.csv', homo, 'homo')
    del homo