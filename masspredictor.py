import tensorflow as tf
from adftPerformance.models.MJ1 import MJ1_Validator, MJ1_Preprocessor, MJ1_Predictor
import pandas as pd

if __name__ == '__main__':
      
    print('Initializing...')
    val = MJ1_Validator(input='smiles')
    prep = MJ1_Preprocessor(optimize=True)

    print('Initializing predictors...')
    homo = MJ1_Predictor(
        model_path='/home/michiel/alchemical-DFT-performance/model_files/homo',
        validator=val,
        preprocessor=prep)
    
    lumo = MJ1_Predictor(
        model_path='/home/michiel/alchemical-DFT-performance/model_files/lumo',
        validator=val,
        preprocessor=prep)
    
    cp = MJ1_Predictor(
        model_path='/home/michiel/alchemical-DFT-performance/model_files/cp',
        validator=val,
        preprocessor=prep)

    ch = MJ1_Predictor(
        model_path='/home/michiel/alchemical-DFT-performance/model_files/ch',
        validator=val,
        preprocessor=prep)

    electro = MJ1_Predictor(
        model_path='/home/michiel/alchemical-DFT-performance/model_files/electro',
        validator=val,
        preprocessor=prep)

    #print('Loading data...')
    #data = pd.read_csv('/home/michiel/alchemical-DFT-performance/dienophile.csv')

    #data = data.iloc[97300:,]

    #smiles = data['smiles'].to_numpy().tolist()
    #print(f'Total smiles to predict: {len(smiles)}')

    #print('=== Predicting homo energies ===')
    #predictions = homo.predict(smiles)
    #data['predicted_homo_ev'] = predictions
    print('HOMO : ', homo.predict('C1=C[C@@H]2C=C[C@@H](S2=O)C=C1'))
    print('LUMO : ', lumo.predict('C1=C[C@@H]2C=C[C@@H](S2=O)C=C1'))
    print('cp : ', cp.predict('C1=C[C@@H]2C=C[C@@H](S2=O)C=C1'))
    print('ch : ', ch.predict('C1=C[C@@H]2C=C[C@@H](S2=O)C=C1'))
    print('electro : ', electro.predict('C1=C[C@@H]2C=C[C@@H](S2=O)C=C1'))
    
    
    
    #print('=== Predicting lumo energies ===')

    #data['predicted_lumo_ev'] = lumo.predict(smiles)
    #print('=== Predicting electrophilicity_index ===')
    
    #data['predicted_electrophilicity_index_ev'] = electro.predict(smiles)

    #data.to_csv('predictions_dienophiles.csv', index=False)