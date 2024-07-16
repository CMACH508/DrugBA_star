import pickle
from model import NetModule
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from molecule import Molecule
import time


class BeamSearch:
    def __init__(self, env, nnet, device, beamsize):
        self.env = env
        self.nnet = nnet
        self.device = device
        self.open = self.env.atom_types
        self.beam_size = beamsize

    def batch_smiles_to_fingerprints(self, smiles):
        fps = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            onbits = list(fp.GetOnBits())
            arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
            arr[onbits] = 1
            fps.append(arr)
        return np.array(fps)

    def estimate_Q_value(self, fps):
        fps_tensor = torch.tensor(fps, device=self.device)
        Q_values = self.nnet(fps_tensor).cpu().data.numpy().reshape(-1)
        return Q_values

    def heuristic_fn(self, states):
        fps = self.batch_smiles_to_fingerprints(states)
        estimated_Qs = self.estimate_Q_value(fps)
        return estimated_Qs

    def step(self):
        child_states = []
        for state in self.open:
            child_states += list(self.env.get_valid_actions(state))
        child_states = list(set(child_states))
        if len(child_states) > self.beam_size:
            batch_size = 102800
            num_batch = len(child_states) // batch_size
            if num_batch * batch_size < len(child_states):
                num_batch += 1
            path_costs = []
            heuristics = []
            for i in range(num_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(child_states))
                path_costs += list(self.env.batch_reward(child_states[start: end]))
                heuristics += list(self.heuristic_fn(child_states[start: end]))
            cost = np.array(path_costs) + np.array(heuristics)
            moves = np.argsort(cost)[-self.beam_size:]
            child_states = [child_states[move] for move in moves]
        self.open = child_states

    def search(self, epoch):
        for i in range(38):
            print(i)
            self.step()
        child_states = []
        for state in self.open:
            child_states += list(self.env.get_valid_actions(state))
        self.open = list(set(child_states))
        cost = list(self.env.batch_reward(self.open))
        moves = np.argsort(cost)[-1000:]
        cost = [cost[move] for move in moves]
        smiles = [self.open[move] for move in moves]
        fr = open('beam_result_model.txt', 'a')
        fr.write(str(np.mean(cost)) + '\n')
        fr.close()
        ans = {
            'smiles': smiles,
            'score': cost
        }
        with open('./test_result/result_beam_model_{:d}_{:d}.pkl'.format(self.beam_size, epoch), 'wb') as writer:
            pickle.dump(ans, writer, protocol=4)



if __name__ == "__main__":
    device = 'cuda:0'
    hidden_dims = [1024, 512, 128, 32]
    input_dim = 2048
    env = Molecule(
        init_mol=None,
        atom_types=['C', 'N', 'O'],
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=True,
        allow_ring_sizes=[3, 4, 5, 6],
        max_steps=40
    )
    for beamsize in [100, 1000]:
        for epoch in [14]:
            parameters = torch.load('./model/model_{:d}.model'.format(epoch), map_location={'cuda:0': device})
            nnet = NetModule(input_dim=2048, hidden_dims=[1024, 512, 128, 32]).to(device)
            nnet.load_state_dict(parameters)
            player = BeamSearch(env, nnet, device, beamsize)
            start = time.time()
            player.search(epoch)
            end = time.time()
            print(end - start)
