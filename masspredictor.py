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
    
    electro = MJ1_Predictor(
        model_path='/home/michiel/alchemical-DFT-performance/model_files/electro',
        validator=val,
        preprocessor=prep)

    print('Loading data...')
    data = pd.read_csv('/home/michiel/alchemical-DFT-performance/dienes.csv')

    

    data.to_csv('test.csv', index=False)