import os
import tensorflow as tf
from adftPerformance.models.MJ1 import MJ1_Validator, MJ1_Preprocessor, MJ1_Predictor

def main():

    val = MJ1_Validator(input='smiles')
    prep = MJ1_Preprocessor(optimize=True)

    pred = MJ1_Predictor(
        model_path='/home/michiel/Paper-Electrophilic-AI/models/electrophilicity_index.tf',
        validator=val,
        preprocessor=prep)
    
    prediction = pred.predict(['C(=CC=O)C=C(C(=O)O)N', 'CC1=CC(=NC1)OC2CC2'])
    print('PREDICTIONS: ', prediction, ' eV')

    print(pred.predict('CC(C)(C)C=CC(C)(C)C'))

if __name__ == '__main__':
    main()