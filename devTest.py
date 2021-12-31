import os
import tensorflow as tf
from adftPerformance.models.MJ1 import MJ1_Validator, MJ1_Preprocessor, MJ1_Predictor

def main():

    val = MJ1_Validator(input='smiles')
    prep = MJ1_Preprocessor(optimize=True)

    pred = MJ1_Predictor(
        model_path='/home/michiel/projects/alchemical-DFT-performance/tests/models/chem_pot/chem_pot',
        validator=val,
        preprocessor=prep)
    
    prediction = pred.predict('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
    print('PREDICTION: ', prediction, ' eV')

if __name__ == '__main__':
    main()