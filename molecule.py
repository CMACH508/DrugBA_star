import collections
import copy
import subprocess
import itertools
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer
from six.moves import range
from six.moves import zip
import os
import pickle
import gzip
import networkx as nx
import math
import time
_fscores = None


def atom_valences_max(atom_types):
    """
    Creates a list of valences corresponding to atom_types.
    Note that this is not a count of valence electrons, but a count of the
    maximum number of bonds each element will make. For example, passing
    atom_types ['C', 'H', 'O'] will return [4, 1, 2].
    :param atom_types: List of string atom types, e.g. ['C', 'H', 'O'].
    :return: List of integer atom valences.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """
    Computes the Bemis-Murcko scaffold for a molecule.
    :param mol: RDKit Mol.
    :return: String scaffold SMILES.
    """
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """
    Returns whether mol contains the given scaffold.
    NOTE: This is more advanced than simply computing scaffold equality (i.e.
    scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
    be a subset of the (possibly larger) scaffold in mol.
    :param mol: RDKit Mol.
    :param scaffold: String scaffold SMILES.
    :return: Boolean whether scaffold is found in mol.
    """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """
    Calculates the largest ring size in the molecule.
    :param molecule: Chem.Mol. A molecule.
    :return: Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """
    Calculates the penalized logP of a molecule.
    Penalized logP is defined as:
    y(m) = logP(m) - SA(m) - cycle(m)
    y(m) is the penalized logP,
    logP(m) is the logP of a molecule,
    SA(m) is the synthetic accessibility score,
    cycle(m) is the largest ring size minus by six in the molecule.
    :param molecule: Chem.Mol. A molecule.
    :return: Float. The penalized logP value.
    """
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score


def penalized_logp_norm(molecule):
    # Modified from https://github.com/bowenliu16/rl_graph_generation
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455
    log_p = Descriptors.MolLogP(molecule)
    SA = - sascorer.calculateScore(molecule)
    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(molecule)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle


def mw(mol):
    '''
    molecular weight estimation from qed
    '''
    x = Descriptors.MolWt(mol)
    a, b, c, d, e, f = 2.817, 392.575, 290.749, 2.420, 49.223, 65.371
    g = math.exp(-(x - c + d/2) / e)
    h = math.exp(-(x - c - d/2) / f)
    x = a + b / (1 + g) * (1 - 1 / (1 + h))
    return x / 104.981


def logp_score(molecule):
    """
    Calculates the logP of a molecule.
    :param molecule: Chem.Mol. A molecule.
    :return: Float. The logP value
    """
    return Descriptors.MolLogP(molecule)


def sa_score(molecule):
    """
    Calculates the SA score of a molecule.
    :param molecule: Chem.Mol. A molecule.
    :return: Float. The SA Score
    """
    from rdkit.Contrib.SA_Score import sascorer
    return sascorer.calculateScore(molecule)


def QED_score(molecule):
    """
    Calculte the QED score of a molecule
    :param molecule: Chem.Mol. A molecule
    :return: Float. The QED Score
    """
    return QED.qed(molecule)


def readNPModel(filename=None):
    """Reads and returns the scoring model,
    which has to be passed to the scoring functions."""
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), 'publicnp.model.gz')
    global _fscores
    _fscores = pickle.load(gzip.open(filename))
    return _fscores


def calScores(molecule):
    """
    Calculate MolLogP, QED, SA Score, NP score, Docking Score of a molecule
    :param molecule: Chem.Mol. A molecule
    :return: List of all scores
    """
    des_list = ['MolLogP', 'qed', 'TPSA', 'MolWt']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    MolLogP, QED, tpsa, MolWt = calculator.CalcDescriptors(molecule)
    sa_score = sascorer.calculateScore(molecule)
    np_score = npscorer.scoreMol(molecule)
    return MolLogP, QED, tpsa, MolWt, sa_score, np_score


def CaculateAffinity(smi, file_protein='./1zys.pdb', file_lig_ref='./1zys_D_199.sdf', out_path='./', prefix=''):
    try:
        mol = Chem.MolFromSmiles(smi)
        m2 = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2)
        m3 = Chem.RemoveHs(m2)
        file_output = os.path.join(out_path, prefix + str(time.time()) + '.pdb')
        Chem.MolToPDBFile(m3, file_output)
        # mol = Chem.MolFromPDBFile("test.pdb")
        # smile = Chem.MolToSmiles(mol)
        # logger.info(smile)
        # logger.info(smi)
        # file_drug="sdf_ligand_"+str(pdb_id)+str(i)+".sdf"
        smina_cmd_output = os.path.join(out_path, prefix + str(time.time()))
        launch_args = ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand",
                       file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9", ">>",
                       smina_cmd_output]
        # launch_args = ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand",
        #             file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9","-o", prefix+'dockres.pdb']
        # -o 1OYT-redock.pdbqt
        launch_string = ' '.join(launch_args)
        p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
        p.communicate()
        affinity = 500
        with open(smina_cmd_output, 'r') as f:
            for lines in f.readlines():
                lines = lines.split()
                if len(lines) == 4 and lines[0] == '1':
                    affinity = float(lines[1])
        p = subprocess.Popen('rm -rf ' + smina_cmd_output, shell=True, stdout=subprocess.PIPE)
        p.communicate()
        p = subprocess.Popen('rm -rf ' + file_output, shell=True, stdout=subprocess.PIPE)
        p.communicate()
    except:
        affinity = 500
    return affinity


class Result(collections.namedtuple('Result', ['state', 'reward', 'terminated'])):
    """
    A namedtuple defines the result of a step for the molecule class.

    The namedtuple contains the following fields:
      state: Chem.RWMol. The molecule reached after taking the action.
      reward: Float. The reward get after taking the action.
      terminated: Boolean. Whether this episode is terminated.
    """


def get_valid_actions(state, atom_types, allow_removal, allow_no_modification, allowed_ring_sizes, allow_bonds_between_rings):
    """
    Computes the set of valid actions for a given state.
    :param state: String SMILES; the current state. If None or the empty string, we
      assume an "empty" state with no atoms or bonds.
    :param atom_types: Set of string atom types, e.g. {'C', 'O'}.
    :param allow_removal: Boolean whether to allow actions that remove atoms and bonds.
    :param allow_no_modification: Boolean whether to include a "no-op" action.
    :param allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    :param allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.
    :return: Set of string SMILES containing the valid actions (technically, the set of
      all states that are acceptable from the given state).
    Raises:
      ValueError: If state does not represent a valid molecule.
    """
    if not state:
        # Available actions are adding a node of each type.
        return copy.deepcopy(atom_types)
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError('Received invalid state: %s' % state)
    atom_valences = {atom_type: atom_valences_max([atom_type])[0] for atom_type in atom_types}
    atoms_with_free_valence = {}
    for i in range(1, max(atom_valences.values())):
        # Only atoms that allow us to replace at least one H with a new bond are enumerated here.
        atoms_with_free_valence[i] = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetNumImplicitHs() >= i]
    valid_actions = set()
    valid_actions.update(_atom_addition(mol, atom_types=atom_types, atom_valences=atom_valences, atoms_with_free_valence=atoms_with_free_valence))
    valid_actions.update(_bond_addition(mol, atoms_with_free_valence=atoms_with_free_valence, allowed_ring_sizes=allowed_ring_sizes, allow_bonds_between_rings=allow_bonds_between_rings))
    if allow_removal:
        valid_actions.update(_bond_removal(mol))
    if allow_no_modification:
        valid_actions.add(Chem.MolToSmiles(mol))
    return list(valid_actions)


def _atom_addition(state, atom_types, atom_valences, atoms_with_free_valence):
    """
    Computes valid actions that involve adding atoms to the graph.
    Actions:
      * Add atom (with a bond connecting it to the existing graph)
      Each added atom is connected to the graph by a bond. There is a separate
      action for connecting to (a) each existing atom with (b) each valence-allowed
      bond type. Note that the connecting bond is only of type single, double, or
      triple (no aromatic bonds are added).
      For example, if an existing carbon atom has two empty valence positions and
      the available atom types are {'C', 'O'}, this section will produce new states
      where the existing carbon is connected to (1) another carbon by a double bond,
      (2) another carbon by a single bond, (3) an oxygen by a double bond, and
      (4) an oxygen by a single bond.
    :param state: RDKit Mol.
    :param atom_types: Set of string atom types.
    :param atom_valences: Dict mapping string atom types to integer valences.
    :param atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.
    :return: Set of string SMILES; the available actions.
    """
    bond_order = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    atom_addition = set()
    for i in bond_order:
        for atom in atoms_with_free_valence[i]:
            for element in atom_types:
                if atom_valences[element] >= i:
                    new_state = Chem.RWMol(state)
                    idx = new_state.AddAtom(Chem.Atom(element))
                    new_state.AddBond(atom, idx, bond_order[i])
                    sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    atom_addition.add(Chem.MolToSmiles(new_state))
    return atom_addition


def _bond_addition(state, atoms_with_free_valence, allowed_ring_sizes, allow_bonds_between_rings):
    """
    Computes valid actions that involve adding bonds to the graph.
    Actions (where allowed):
      * None->{single,double,triple}
      * single->{double,triple}
      * double->{triple}
    :param state: RDKit Mol.
    :param atoms_with_free_valence: Dict mapping integer minimum available valence
      values to lists of integer atom indices. For instance, all atom indices in
      atoms_with_free_valence[2] have at least two available valence positions.
    :param allowed_ring_sizes: Set of integer allowed ring sizes; used to remove some
      actions that would create rings with disallowed sizes.
    :param allow_bonds_between_rings: Boolean whether to allow actions that add bonds
      between atoms that are both in rings.
    :return: Set of string SMILES; the available actions.
    """
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]
    bond_addition = set()
    for valence, atoms in atoms_with_free_valence.items():
        for atom1, atom2 in itertools.combinations(atoms, 2):
            # Get the bond from a copy of the molecule so that SetBondType() doesn't
            # modify the original state.
            bond = Chem.Mol(state).GetBondBetweenAtoms(atom1, atom2)
            new_state = Chem.RWMol(state)
            # Kekulize the new state to avoid sanitization errors; note that bonds
            # that are aromatic in the original state are not modified (this is
            # enforced by getting the bond from the original state with
            # GetBondBetweenAtoms()).
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            if bond is not None:
                if bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.
                idx = bond.GetIdx()
                # Compute the new bond order as an offset from the current bond order.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order += valence
                if bond_order < len(bond_orders):
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state.ReplaceBond(idx, bond)
                else:
                    continue
            # If do not allow new bonds between atoms already in rings.
            elif (not allow_bonds_between_rings and
                  (state.GetAtomWithIdx(atom1).IsInRing() and
                   state.GetAtomWithIdx(atom2).IsInRing())):
                continue
            # If the distance between the current two atoms is not in the
            # allowed ring sizes
            elif (allowed_ring_sizes is not None and
                  len(Chem.rdmolops.GetShortestPath(
                      state, atom1, atom2)) not in allowed_ring_sizes):
                continue
            else:
                new_state.AddBond(atom1, atom2, bond_orders[valence])
            sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
            # When sanitization fails
            if sanitization_result:
                continue
            bond_addition.add(Chem.MolToSmiles(new_state))
    return bond_addition


def _bond_removal(state):
    """
    Computes valid actions that involve removing bonds from the graph.
    Actions (where allowed):
      * triple->{double,single,None}
      * double->{single,None}
      * single->{None}
    Bonds are only removed (single->None) if the resulting graph has zero or one
    disconnected atom(s); the creation of multi-atom disconnected fragments is not
    allowed. Note that aromatic bonds are not modified.
    :param state: RDKit Mol.
    :return: Set of string SMILES; the available actions.
    """
    bond_orders = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
    ]
    bond_removal = set()
    for valence in [1, 2, 3]:
        for bond in state.GetBonds():
            # Get the bond from a copy of the molecule so that SetBondType() doesn't
            # modify the original state.
            bond = Chem.Mol(state).GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                                       bond.GetEndAtomIdx())
            if bond.GetBondType() not in bond_orders:
                continue  # Skip aromatic bonds.
            new_state = Chem.RWMol(state)
            # Kekulize the new state to avoid sanitization errors; note that bonds
            # that are aromatic in the original state are not modified (this is
            # enforced by getting the bond from the original state with
            # GetBondBetweenAtoms()).
            Chem.Kekulize(new_state, clearAromaticFlags=True)
            # Compute the new bond order as an offset from the current bond order.
            bond_order = bond_orders.index(bond.GetBondType())
            bond_order -= valence
            if bond_order > 0:  # Downgrade this bond.
                idx = bond.GetIdx()
                bond.SetBondType(bond_orders[bond_order])
                new_state.ReplaceBond(idx, bond)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When sanitization fails
                if sanitization_result:
                    continue
                bond_removal.add(Chem.MolToSmiles(new_state))
            elif bond_order == 0:  # Remove this bond entirely.
                atom1 = bond.GetBeginAtom().GetIdx()
                atom2 = bond.GetEndAtom().GetIdx()
                new_state.RemoveBond(atom1, atom2)
                sanitization_result = Chem.SanitizeMol(new_state, catchErrors=True)
                # When sanitization fails
                if sanitization_result:
                    continue
                smiles = Chem.MolToSmiles(new_state)
                parts = sorted(smiles.split('.'), key=len)
                # We define the valid bond removing action set as the actions
                # that remove an existing bond, generating only one independent
                # molecule, or a molecule and an atom.
                if len(parts) == 1 or len(parts[0]) == 1:
                    bond_removal.add(parts[-1])
    return bond_removal


class Molecule(object):
    """
    Defines the Markov decision process of generating a molecule.
    """
    def __init__(self, atom_types, init_mol=None, allow_removal=True, allow_no_modification=True, allow_bonds_between_rings=True,
                 allow_ring_sizes=None, max_steps=40, target_fn=None, record_path=False):
        """
        Initializes the parameters for the MDP.
        Internal state will be stored as SMILES strings
        :param atom_types: The set of elements the molecule may contain.
        :param init_mol: String, Chem.Mol or Chem.RWMol. If string is provided, it is considered as the SMILES string.
          The molecule to be set as the initial state. If None, an empty molecule will be created
        :param allow_removal: Boolean. Whether to allow removal of a bond.
        :param allow_no_modification: Boolean. If true, the valid action set will include doing nothing to the current
          molecule, i.e., the current molecule itself will be added to the action set.
        :param allow_bonds_between_rings: Boolean. If False, new bonds connecting two atoms which are both in rings are not allowed.
          DANGER Set this to False will disable some of the transformations eg. c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
          But it will make the molecules generated make more sense chemically.
        :param allow_ring_sizes: Set of integers or None. The size of the ring which is allowed to form. If None,
          all sizes will be allowed. If a set is provided, only sizes in the set is allowed.
        :param max_steps: Integer. The maximum number of steps to run.
        :param target_fn: A function or None. The function should have Args of a String, which is a SMILES string (the state),
          and Returns as a Boolean which indicates whether the input satisfies a criterion. If None, it will not be used as a criterion.
        :param record_path: Boolean. Whether to record the steps internally.
        """
        if isinstance(init_mol, Chem.Mol):
            init_mol = Chem.MolToSmiles(init_mol)
        self.init_mol = init_mol
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allow_ring_sizes
        self.max_steps = max_steps
        self._state = None
        self._valid_actions = []
        self._counter = self.max_steps
        self._target_fn = target_fn
        self.record_path = record_path
        self._path = []
        self.is_terminated = False
        self._max_bonds = 4
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(list(zip(atom_types, atom_valences_max(atom_types))))

    @property
    def state(self):
        return self._state

    @property
    def num_steps_taken(self):
        return self._counter

    def get_path(self):
        return self._path

    def initialize(self):
        """
        Resets the MDP to its initial state
        """
        self._state = self.init_mol
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0

    def get_valid_actions(self, state=None, force_rebuild=False):
        """
        Gets the valid actions for the state.
        In this design, we do not further modify a aromatic ring. For example,
        we do not change a benzene to a 1,3-Cyclohexadiene. That is, aromatic
        bonds are not modified.
        :param state: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
        considered as the SMILES string. The state to query. If None, the current
        state will be considered.
        :param force_rebuild: Boolean. Whether to force rebuild of the valid action set.
        :return: A set contains all the valid actions for the state. Each action is a
        SMILES string. The action is actually the resulting state.
        """
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        if isinstance(state, Chem.Mol):
            state = Chem.MolToSmiles(state)
        valid_actions = list(get_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings))
        self._valid_actions = []
        for action in valid_actions:
            try:
                mol = Chem.MolFromSmiles(action)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self._valid_actions.append(action)
            except:
                continue
        return copy.deepcopy(self._valid_actions)

    def _reward(self):
        """
        Gets the reward for the state. A child class can redefine the reward function if reward other than zero is desired.
        :return: Float. The reward for the current state.
        """
        mol = Chem.MolFromSmiles(self._state)
        QED_score = QED.qed(mol)
        SA_score = - sascorer.calculateScore(mol)
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        SA_score = (SA_score - SA_mean) / SA_std
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        log_p = Descriptors.MolLogP(mol)
        log_p = (log_p - logP_mean) / logP_std
        return QED_score + 0.2 * SA_score

    def batch_reward(self, smiles):
        ans = []
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        for a in smiles:
            mol = Chem.MolFromSmiles(a)
            QED_score = QED.qed(mol)
            SA_score = - sascorer.calculateScore(mol)
            SA_score = (SA_score - SA_mean) / SA_std
            log_p = Descriptors.MolLogP(mol)
            log_p = (log_p - logP_mean) / logP_std
            ans.append(QED_score + 0.2 * SA_score)
        return ans

    def _goal_reached(self):
        """
        Sets the termination criterion for molecule Generation. A child class can define this function to terminate the MDP before max_steps is reached.
        :return:  Boolean, whether the goal is reached or not. If the goal is reached, the MDP is terminated.
        """
        if self._target_fn is None:
            return False
        return self._target_fn(self._state)

    def step(self, action):
        """
        Takes a step forward according to the action.
        :param action: Chem.RWMol. The action is actually the target of the modification.
        :return: results: Namedtuple containing the following fields:
          * state: The molecule reached after taking the action.
          * reward: The reward get after taking the action.
          * terminated: Whether this episode is terminated.
        Raises:
          ValueError: If the number of steps taken exceeds the preset max_steps, or
          the action is not in the set of valid_actions.
        """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError('This episode is terminated.')
        if action not in self._valid_actions:
            raise ValueError('Invalid action.')
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1
        result = Result(
            state=self._state,
            reward=self._reward(),
            terminated=(self._counter >= self.max_steps) or self._goal_reached())
        return result

    def visualize_state(self, state=None, **kwargs):
        """
        Draws the molecule of the state.
        :param state:  String, Chem.Mol, or Chem.RWMol. If string is prov ided, it is
        considered as the SMILES string. The state to query. If None, the current state will be considered.
        :param kwargs:The keyword arguments passed to Draw.MolToImage.
        :return: A PIL image containing a drawing of the molecule.
        """
        if state is None:
            state = self._state
        if isinstance(state, str):
            state = Chem.MolFromSmiles(state)
        return Draw.MolToImage(state, **kwargs)
