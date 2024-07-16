import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from molecule import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import pickle
import os
import glob
from schedules import PiecewiseSchedule
from multiprocessing import Process
import multiprocessing
import time


class NetModule(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(NetModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                block_fc = nn.Linear(self.input_dim, self.hidden_dims[i])
            else:
                block_fc = nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i])
            block_bn = nn.BatchNorm1d(self.hidden_dims[i])
            self.layers.append(nn.ModuleList([block_fc, block_bn]))
        self.fc_out = nn.Linear(self.hidden_dims[-1], 1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i][1](self.layers[i][0](x)))
        x = F.relu(self.fc_out(x))
        return x


def flatten(data):
    num_each = [len(x) for x in data]
    split_idxs = list(np.cumsum(num_each)[:-1])
    data_flat = [item for sublist in data for item in sublist]
    return data_flat, split_idxs


def unflatten(data, split_idxs):
    data_split = []
    start_idx = 0
    for end_idx in split_idxs:
        data_split.append(data[start_idx: end_idx])
        start_idx = end_idx
    data_split.append(data[start_idx:])
    return data_split


class SynDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['state'])

    def __getitem__(self, item):
        state = self.data['state'][item]
        g_state = self.data['g_state'][item]
        next_state = self.data['next_state'][item]
        g_next = self.data['g_next'][item]
        next_terminated = self.data['next_terminated'][item]
        return state, g_state, next_state, g_next, next_terminated


class DDQN():
    def __init__(self, update=False, gpu=0):
        self.hidden_dims = [1024, 512, 128, 32]
        self.input_dim = 2048
        self.device = 'cuda:' + str(gpu)
        self.eval_Q = NetModule(input_dim=self.input_dim, hidden_dims=self.hidden_dims)
        self.eval_Q = self.eval_Q.to(self.device)
        if update:
            self.target_Q = NetModule(input_dim=self.input_dim, hidden_dims=self.hidden_dims)
            self.target_Q = self.target_Q.to(self.device)
        self.target_network_update_freq = 100
        self.minibatch_size = 128
        self.optimizer = optim.Adam(self.eval_Q.parameters(), lr=0.01, weight_decay=0.00001)
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.atom_types = ['C', 'N', 'O']
        self.max_steps = 40
        self.epoch = 0
        self.training_steps = 0
        self.max_episode = 100
        self.epsilon_schedule = PiecewiseSchedule([(0, 1.0), (int(self.max_episode/2), 0.1), (self.max_episode, 0.01)], outside_value=0.01)
        self.env = Molecule(
            atom_types=self.atom_types,
            allow_removal=True,
            allow_no_modification=True,
            allow_bonds_between_rings=True,
            allow_ring_sizes=[3, 4, 5, 6],
            max_steps=self.max_steps
        )
        self.env.initialize()

    def batch_smiles_to_fingerprints(self, smiles):
        fps = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.input_dim)
            onbits = list(fp.GetOnBits())
            arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
            arr[onbits] = 1
            fps.append(arr)
        return np.array(fps)

    def estimate_Q_value(self, fps):
        fps_tensor = torch.tensor(fps, device=self.device)
        Q_values = self.eval_Q(fps_tensor).cpu().data.numpy().reshape(-1)
        return Q_values

    def estimate_Q_value_Batch(self, fps):
        num_batch = int(len(fps) // self.minibatch_size)
        if num_batch * self.minibatch_size < len(fps):
            num_batch += 1
        Q_values = np.array([])
        for i in range(num_batch):
            start = i * self.minibatch_size
            end = min((i + 1) * self.minibatch_size, len(fps))
            fps_tensor = torch.tensor(np.array(fps[start: end]), device=self.device)
            Q_values = np.append(Q_values, self.eval_Q(fps_tensor).cpu().data.numpy().reshape(-1))
        return Q_values

    def samples_a_action(self, f_values, sample=False, epsilon=True, epsilon_value=None):
        if sample:
            pdf = np.exp(f_values)
            cdf = pdf.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            move = cdf.searchsorted(selection)
        elif epsilon:
            if np.random.uniform() > epsilon_value:
                #move = np.argsort(f_values)[-3:]
                #move = np.random.choice(move)
                move = np.argmax(f_values)
            else:
                move = np.random.choice([i for i in range(len(f_values))])
        else:
            move = np.argmax(f_values)
        return move

    def play_a_game_test(self, init_mol):
        self.eval_Q.eval()
        env = Molecule(
            init_mol=init_mol,
            atom_types=self.atom_types,
            allow_removal=True,
            allow_no_modification=True,
            allow_bonds_between_rings=True,
            allow_ring_sizes=[3, 4, 5, 6],
            max_steps=self.max_steps
        )
        env.initialize()
        for step in range(self.max_steps):
            fps = self.batch_smiles_to_fingerprints(env._valid_actions)
            g_values = env.batch_reward(env._valid_actions)
            Q_values = self.estimate_Q_value(fps)
            f_values = g_values + Q_values
            move = self.samples_a_action(f_values, sample=True, epsilon=False, epsilon_value=0.0)
            result = env.step(env._valid_actions[move])
        return result.state, result.reward

    def play_games_test(self, init_mols):
        self.eval_Q.eval()
        num_envs = len(init_mols)
        envs = []
        for i in range(num_envs):
            env = Molecule(
                init_mol=init_mols[i],
                atom_types=self.atom_types,
                allow_removal=True,
                allow_no_modification=True,
                allow_bonds_between_rings=True,
                allow_ring_sizes=[3, 4, 5, 6],
                max_steps=self.max_steps
            )
            env.initialize()
            envs.append(env)
        for step in range(self.max_steps):
            fps = []
            g_values = []
            for i in range(num_envs):
                fps.append(self.batch_smiles_to_fingerprints(envs[i]._valid_actions))
                g_values.append(envs[i].batch_reward(envs[i]._valid_actions))
            fps, split_idxs = flatten(fps)
            Q_values = self.estimate_Q_value(fps)
            Q_values = unflatten(Q_values, split_idxs)
            results = []
            for i in range(num_envs):
                move = self.samples_a_action(g_values[i] + Q_values[i], sample=True, epsilon=False, epsilon_value=0.0)
                results.append(envs[i].step(envs[i]._valid_actions[move]))
        ans = {}
        for i in range(num_envs):
            if results[i].state not in ans.keys():
                ans[results[i].state] = []
            ans[results[i].state].append(results[i].reward)
        return ans

    def collect_games(self, num_games, epoch, thread):
        self.eval_Q.eval()
        envs = []
        data = [[] for _ in range(num_games)]
        for i in range(num_games):
            env = Molecule(
                atom_types=self.atom_types,
                allow_removal=True,
                allow_no_modification=True,
                allow_bonds_between_rings=True,
                allow_ring_sizes=[3, 4, 5, 6],
                max_steps=self.max_steps
            )
            env.initialize()
            envs.append(env)
        for step in range(self.max_steps):
            fps = []
            g_values = []
            for i in range(num_games):
                fps.append(self.batch_smiles_to_fingerprints(envs[i]._valid_actions))
                g_values.append(envs[i].batch_reward(envs[i]._valid_actions))
            fps, split_idxs = flatten(fps)
            Q_values = self.estimate_Q_value_Batch(fps)
            Q_values = unflatten(Q_values, split_idxs)
            for i in range(num_games):
                move = self.samples_a_action(g_values[i] + Q_values[i], sample=True, epsilon=False, epsilon_value=0.1)
                result = envs[i].step(envs[i]._valid_actions[move])
                data[i].append([result.state, result.reward])
        data_processed = []
        for i in range(num_games):
            for j in range(len(data[i]) - 2):
                data_processed.append([data[i][j][0], data[i][j][1], data[i][j + 1][0], data[i][j + 1][1], False])
            data_processed.append([data[i][-2][0], data[i][-2][1], data[i][-1][0], data[i][-1][1], True])
        file_name = './data/data_' + str(epoch) + '_' + str(thread) + '.pkl'
        with open(file_name, 'wb') as writer:
            pickle.dump(data_processed, writer, protocol=4)

    def prepare_data(self):
        file_names = glob.glob('./data/*.pkl')
        data = {
            'state': [],
            'g_state': [],
            'next_state': [],
            'g_next': [],
            'next_terminated': []
        }
        for file_name in file_names:
            with open(file_name, 'rb') as f:
                current = pickle.load(f)
            data['state'] += [current[i][0] for i in range(len(current))]
            data['g_state'] += [current[i][1] for i in range(len(current))]
            data['next_state'] += [current[i][2] for i in range(len(current))]
            data['g_next'] += [current[i][3] for i in range(len(current))]
            data['next_terminated'] += [current[i][4] for i in range(len(current))]
            os.remove(file_name)
        file_name = './data_saved/data_' + str(self.epoch) + '.pkl'
        with open(file_name, 'wb') as writer:
            pickle.dump(data, writer, protocol=4)
        return SynDataSet(data)

    def update_a_batch(self, batch):
        states, g_states, next_states, g_nexts, next_terminated = batch
        self.target_Q.eval()
        next_state_fps = self.batch_smiles_to_fingerprints(next_states)
        next_state_fps_tensor = torch.tensor(next_state_fps, device=self.device)
        next_state_Q_values = self.target_Q(next_state_fps_tensor).cpu().data.numpy().reshape(-1)
        Q_target = []
        for i in range(len(states)):
            if next_terminated[i]:
                f_score = g_nexts[i]
            else:
                f_score = g_nexts[i] + next_state_Q_values[i]
            Q_target.append(f_score - g_states[i])
        self.eval_Q.train()
        state_fps = self.batch_smiles_to_fingerprints(states)
        state_fps_tensor = torch.tensor(state_fps, device=self.device).float()
        Q_target = torch.tensor(Q_target, device=self.device).float()
        Q_estimate = self.eval_Q(state_fps_tensor).reshape(-1)
        loss = self.loss_fn(Q_estimate, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.data.item()
        fr = open('loss.txt', 'a')
        fr.write(str(loss) + str('\n'))
        fr.close()

    def update(self, data):
        self.epoch += 1
        trainLoader = DataLoader(data, batch_size=self.minibatch_size, shuffle=True, num_workers=8)
        for batch in trainLoader:
            self.update_a_batch(batch)
            self.training_steps += 1
            if self.training_steps % self.target_network_update_freq == 0:
                self.target_Q.load_state_dict(self.eval_Q.state_dict())

    def test(self, init_mols):
        result = self.play_games_test(init_mols)
        file = './test/test_' + str(self.epoch) + '.pkl'
        with open(file, 'wb') as writer:
            pickle.dump(result, writer, protocol=4)

    def save_model(self):
        modelName = './model/model_' + str(self.epoch) + '.model'
        torch.save(self.eval_Q.state_dict(), modelName)

    def load_model(self, model_path):
        parameters = torch.load(model_path, map_location={'cuda:0': self.device})
        self.eval_Q.load_state_dict(parameters)


def collect_games(gpu, num_games, epoch, thread):
    model_path = './model/model_' + str(epoch) + '.model'
    player = DDQN(update=False, gpu=gpu)
    player.load_model(model_path)
    player.collect_games(num_games, epoch, thread)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    player_base = DDQN(update=True, gpu=0)
    player_base.save_model()
    num_parallel = 8
    num_games = 125
    gpus = [0]
    while player_base.epoch < 15:
        jobs = [Process(target=collect_games, args=(gpus[0], num_games, player_base.epoch, i)) for i in range(num_parallel)]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()
        data = player_base.prepare_data()
        player_base.update(data)
        player_base.save_model()

