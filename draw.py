from numpy import imag
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

if __name__ == '__main__':

    data = pd.read_csv('/home/michiel/alchemical-DFT-performance/predictions_dienophiles.csv')

    data = data.sort_values(by=['predicted_electrophilicity_index_ev'], ascending=False)
    data = data.loc[data['predicted_electrophilicity_index_ev'] > 4]

    mols = []
    i = 1
    for cid, smiles, homo in zip(data['cid'].tolist(), data['smiles'].tolist(), data['predicted_electrophilicity_index_ev'].tolist()):
        m = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(m)
        m.SetProp('legend', f'{i}\ncid: {cid}\nsmiles: {smiles}\npredicted_electrophilicity_index_ev: {homo}')
        mols.append(m)
        i +=1
    
    image = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(150, 100), legends=[x.GetProp("legend") for x in mols], useSVG=True)
    #image.save('shortlist.png')
    
    with open('shortlist_dienophiles.svg', 'w') as f:
        f.write(image)