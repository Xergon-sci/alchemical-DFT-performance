""" This module contains functionality to predict with the MJ1 Series of models.
"""

from logging import WARNING
import re
import warnings
from adftPerformance.base import Validator, Preprocessor, Predictor
from openbabel.openbabel import OBMol, OBConversion, OBBuilder, OBForceField
from qubit.descriptors import CoulombMatrix
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class MJ1_Validator(Validator):
    """MJ1_Validator validates both smiles and xyz input for prediction with the MJ1 series of models.
    """

    def __init__(self, input='smiles'):
        """__init__

        :param input: input type, can be either 'smiles' or 'xyz', defaults to 'smiles'
        :type input: str, optional
        """
        self.input = input

    def validate(self, molecule):
        """validate Validates the molecule for use with the MJ1 series of models, can be either smiles or a xyz file.

        :param molecule: The molecule to predict.
        :type molecule: str
        :raises ValueError: When the molecule cant load.
        :raises ValueError: When the molecule doesn't contain the right range of heavy atoms.
        :raises ValueError: When the molecule containers an atom other then CNOSH.
        :raises ValueError: When the molecule contains a charge.
        :return: An OBMOL object, that contains the molecule.
        :rtype: OBMOL
        """
        if self.input == 'smiles':
            obconversion = OBConversion()
            obconversion.SetInFormat('smi')

            mol = OBMol()
            obconversion.ReadString(mol, molecule)
        elif self.input == 'xyz':
            obconversion = OBConversion()
            obconversion.SetInFormat('xyz')

            mol = OBMol()
            obconversion.ReadString(mol, molecule)
        else:
            raise ValueError
        
        #if mol.NumAtoms() <= 0:
        #    warnings.warn(f'No atoms found, are {molecule} valid smiles?')
        #    return None

        #temp_mol = mol
        #if temp_mol.DeleteHydrogens():
        #    heavy_atoms = temp_mol.NumAtoms()
        #    if heavy_atoms is not 10:
        #        warnings.warn(f'Your molecule contains {heavy_atoms} heavy atoms, the predictive range of the MJ1 series is [10] heavy atoms.')
        #        return None

        #    formula = temp_mol.GetSpacedFormula()
        #    formula = re.sub(r'[0-9]', '', formula)
        #    formula = formula.split()

        #    cnos = ['C', 'H', 'N', 'O', 'S']

        #    for s in formula:
        #        if s not in cnos:
        #            warnings.warn(f'Your molecule contains {s} this is not allowed! The only accepted atoms are of types CNOSH.')
        #            return None
            
            #charge = temp_mol.GetTotalCharge()
            #if charge != 0:
            #    raise ValueError(f'Your molecule contains a charge ({charge}), this is not supported.')
        
        return mol

class MJ1_Preprocessor(Preprocessor):
    """MJ1_Preprocessor Preprocesses all the date for the MJ1 series of models.
    """

    def __init__(self, optimize=False, gradients=100):
        super().__init__()
        self.optim = optimize
        self.gradients=gradients
    
    def optimize(self, mol):
        """optimize the geometry by forcefield optimization. The used forcefield is MMFF94.

        :param mol: The molecule that will be optimized.
        :type mol: OBMol
        :param gradients: The amount of gradient updates the forcefield is allowd to calculate, defaults to 100
        :type gradients: int, optional
        :raises ValueError: When the forcefield cannot be establised.
        :return: The FF optimized geometry.
        :rtype: OBMol
        """
        builder = OBBuilder()
        forcefield = OBForceField.FindForceField('MMFF94')

        mol.AddHydrogens()
        builder.Build(mol)

        if forcefield.Setup(mol) is False:
            warnings.warn('Forcefield setup failed, no forcefield available.')
            return None
        else:
            forcefield.ConjugateGradients(self.gradients)
            forcefield.GetCoordinates(mol)
            return mol
    
    def preprocess(self, molecule):
        """preprocess Preprocess the molecule for the MJ1 series of models.

        :param molecule: The molecule to preprocess.
        :type molecule: OBMol
        :param optimize: Optional forcefield optimization, defaults to False
        :type optimize: bool, optional
        :param gradients: The ammount of gradient calculations allowed, defaults to 100
        :type gradients: int, optional
        :return: A tensor representation of the molecule.
        :rtype: np.array
        """

        if self.optim:
            mol = self.optimize(molecule)
        else:
            mol = molecule
        
        obconversion = OBConversion()
        obconversion.SetOutFormat('xyz')
        xyz = obconversion.WriteString(mol, True)
        xyz = '\n'.join(xyz.split('\n')[2:])
        coords = []
        atoms = []
        for l in xyz.splitlines():
            i = l.split()
            atoms.append(i[0])
            t = []
            t.append(i[1])
            t.append(i[2])
            t.append(i[3])
            coords.append(t)
        
        coords = np.asarray(coords, dtype=np.float32)

        x = CoulombMatrix.generate(atoms=atoms, xyz=coords)
        x = CoulombMatrix.pad_matrix(x, size=33)
        x = CoulombMatrix.normalize(x, negative_dimensions=86, positive_dimensions=14)
        return np.expand_dims(x, axis=3)

class MJ1_Predictor(Predictor):
    """MJ1_Predictor Predicts the molecule with the MJ1 series of models.
    """

    def __init__(self, model_path, validator, preprocessor):
        """__init__

        :param models: The path to the model.
        :type models: str
        :param validator: The validator.
        :type validator: adft-performance.base.Validator
        :param preprocessor: The preprocessor?
        :type preprocessor: adft-performance.base.Preprocessor
        """
        super().__init__(model_path, validator, preprocessor)
        self.model = load_model(self.model_path, compile=False)
    
    def predict(self, molecule):
        """predict Predicts a value from the selected model for this molecule.

        :param molecule: The molecule can be a smiles string or a xyz file.
        :type molecule: str
        :return: predictied value
        :rtype: float
        """

        def generator():
            i = 0
            print(f'{len(molecule)} molecules found.')
            while i < len(molecule):
                # select the data from a np matrix
                x = molecule[i]

                x = self.validator.validate(x)
                if x is None:
                    return None
                tensor = self.preprocessor.preprocess(x)
                if tensor is None:
                    return None
                
                #print(i)
                yield tensor
                i += 1

        if type(molecule) is list:
            dataset = tf.data.Dataset.from_generator(
                    generator=generator,
                    output_signature=(
                        tf.TensorSpec(shape=(101,33,33,1), dtype=tf.float32)
                    )
                )

            dataset = dataset.batch(32)

            predictions = self.model.predict(dataset)

            return predictions
        else:
            molecule = self.validator.validate(molecule)
            if molecule is None:
                return None
            tensor = self.preprocessor.preprocess(molecule)
            if tensor is None:
                return None
            tensor = np.expand_dims(tensor, axis=0)

            return self.model.predict(tensor)[0][0]