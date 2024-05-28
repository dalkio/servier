from rdkit.Chem import MolFromSmiles, rdMolDescriptors, rdmolfiles, rdmolops
from rdkit.DataStructs import ExplicitBitVect


def fingerprint_features(smile_string: str, radius: int, size: int) -> ExplicitBitVect:
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=size, useChirality=True, useBondTypes=True, useFeatures=False
    )
