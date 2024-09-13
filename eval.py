from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED, AllChem
from rdkit import Chem, DataStructs
import numpy as np
import pickle
from rdkit.Chem import Descriptors


def validity(smiles_arr):
    success, failure = 0, 0
    for smiles in smiles_arr:
        try:
            mol = Chem.MolFromSmiles(smiles)
            success += 1
        except Exception as e:
            failure += 1
    return success / len(smiles_arr)


def is_success(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    sa_score = sascorer.calculateScore(molecule)
    qed_score = QED.qed(molecule)
    log_p = Descriptors.MolLogP(molecule)
    if qed_score > 0.605 and sa_score < 2.797:
        return True, qed_score, sa_score, log_p
    else:
        return False, qed_score, sa_score, log_p


def success_rate(smiles_arr):
    success = 0
    qed_scores = []
    sa_scores = []
    log_ps = []
    for smiles in smiles_arr:
        succ, qed_score, sa_score, log_p = is_success(smiles)
        qed_scores.append(qed_score)
        sa_scores.append(sa_score)
        log_ps.append(log_p)
        if succ:
            success += 1
    return success / len(smiles_arr), np.mean(qed_scores), np.mean(sa_scores), np.mean(log_ps)


def calcTanimotoSimilarityPairs(smiles1, smiles2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles1), 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles2), 2, nBits=2048)
    return DataStructs.FingerprintSimilarity(fp1, fp2)


def diversity(smiles_arr):
    smiles_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048) for smiles in smiles_arr]
    sum_sim = 0
    for i in range(len(smiles_arr)):
        for j in range(i + 1, len(smiles_arr)):
            sum_sim += 2 * DataStructs.FingerprintSimilarity(smiles_fps[i], smiles_fps[j])
    return 1 - sum_sim / (len(smiles_arr) * (len(smiles_arr) - 1))


def novelty(smiles_arr, smiles_base):
    smiles_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in smiles_arr]
    sims = []
    for fp in smiles_fps:
        for fp_base in smiles_base:
            sims.append(DataStructs.FingerprintSimilarity(fp, fp_base))
    return 1 - np.mean(sims)


if __name__ == "__main__":
    with open('ChemBL.pkl', 'rb') as f:
        smiles_base = pickle.load(f)
    smiles_base = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in smiles_base]
    with open('./test_result/result_beam_model.pkl', 'rb') as f:
        data = pickle.load(f)
    smiles = data['smiles']
    num_mols = len(smiles)
    val_rate = validity(smiles)
    succ_rate, qed_score, sa_score, log_p = success_rate(smiles)
    div = diversity(smiles)
    nov = novelty(smiles, smiles_base)
    result = 'Num: {:d}, Validity: {:f}, Success rate: {:f}, Novelty: {:f}, Diversity: {:f}, QED score: {:f}, SA score: {:f}, logP: {:f}\n'.format(num_mols, val_rate, succ_rate, nov, div, qed_score, sa_score, log_p)
    print(result)


