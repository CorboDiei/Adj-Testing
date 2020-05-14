"""
This module is a modification of the adjacency matrices creating class
that creates adjacency matrices for the valence electrons of each atom in the molecule
It is also updated for Python 3.7

Author: David Corbo
Last Edit: 1/27/20
"""

import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt


class PDBAtom(object):
    """ Class to represent a single atom's position and state at a frame
    
    Attributes:
        _valence_dict (dict{str: int}): A dictionary of valence electron count per element
        x (float): The x coordinate of the atom
        y (float): The y coordinate of the atom
        z (float): The z coordinate of the atom
        valence_count (int): Number of valence electrons in the atom
    """
    
    
    _valence_dict = {'C': 4,
                     'H': 1,
                     'N': 5,
                     'O': 6,
                     'S': 6}
    
    def __init__(self, string):
        """ Standard PDB file format
        ATOM    277  O1  LYS A  14      21.138  -0.865  -4.761  1.00  0.00           O1-
        """
#       Coordinate Parser 
        self.x = float(string[30:38].strip())
        self.y = float(string[38:46].strip())
        self.z = float(string[46:54].strip())
        
#       Element and Valence Electron Number Parser
        self.element_spec = string[77:].strip()
        mod = 0
        if self.element_spec.endswith(('-', '+')):
            self.element_sym = self.element_spec[:-2].strip()
            mod = int(self.element_spec[-2])
            mod *= (-1, 1)[self.element_spec.endswith('-')]
        else:
            self.element_sym = self.element_spec.strip()
        self.valence_count = PDBAtom._valence_dict.get(self.element_sym)
        if self.valence_count is None:
            raise TypeError('Used an element that is not in the valence dictionary')
        else:
            self.valence_count += mod


class Adj_Mats(object):
    """ Class to represent a series of adjacency matrices
    
    Attributes:
        file (str): The path of the pdb file to be parsed
        valence_list (Array[Array[int]]): Stores the number of valence electrons in all atoms in every frame
        distance_graphs (Array[Array[Array[int]]]): The series of distance matrices of the atoms in the evolution
        adjacenecy_graphs (Array[Array[Array[int]]]): The series of adjacency matrices of the atoms in the evolution
        elec_adjacency_graphs (Array[Array[Array[int]]]): The series of adjacency matrices of electrons in the evolution
        
    Methods:
        set_atom_dists: Used to set the distance_graphs attribute
        set_atom_adj: Used to set the adjacency_graphs attribute
        get_atom_dists: Used to parse the pdb file to create a distance_graphs object
        get_atom_adj: Used to set an adjacency threshold on the distance matrices and make adjacency matrices
    """
    
    
    def __init__(self, pdb):
        self.file = pdb
        self.valence_list = np.zeros(1, int)
        self.distance_graphs = np.zeros(1, int)
        self.adjacency_graphs = np.zeros(1, int)
        self.elec_adjacency_graphs = np.zeros(1, int)
        self.eigenvalues = None
        self.bin_probs = None
        self.entropy = None
        self.energy = None
        self.cont_ent = None
    
    def set_atom_dists(self, new_dists):
        self.distance_graphs = new_dists
        
    def set_atom_adj(self, new_adj):
        self.adjacency_graphs = new_adj

    def get_atom_dists(self):
        if os.path.isfile(self.file):
            pdb_file = open(self.file,'r')
        else:
            raise OSError('File {} does not exist'.format(self.file))

        lineno = 0
        frames = []
        atoms = []
        val_frames = []
        val_atoms = []
        
        for line in pdb_file:
            lineno += 1
            if line.startswith('ATOM'):
                try:
                    at_obj = PDBAtom(line)
                    atoms.append([at_obj.x, at_obj.y, at_obj.z])
                    val_atoms.append(at_obj.valence_count)
                except:
                    sys.stderr.write('\nProblem parsing line {} in file {}\n'.format(lineno, self.file))
                    sys.stderr.write(line)
                    sys.stderr.write('Probably ATOM entry is formatted incorrectly?\n')
                    sys.stderr.write('Please refer to - http://www.wwpdb.org/documentation/format32/sect9.html#ATOM\n\n')
                    sys.exit(1)
            elif line.startswith('END'):
                frames.append(atoms)
                atoms = []
                val_frames.append(val_atoms)
                val_atoms = []
        pdb_file.close()
        
        base = np.zeros((len(frames), len(frames[0]), 3))
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                for k in range(len(frames[i][j])):
                    base[i][j][k] = frames[i][j][k]
        dists = np.reshape(base, (len(frames), 1, len(frames[0]), 3)) - np.reshape(base, (len(frames), len(frames[0]), 1, 3))
        dists = dists**2
        dists = dists.sum(3)
        dists = np.sqrt(dists)
        
        self.valence_list = val_frames
        self.distance_graphs = dists
        
        return self.distance_graphs
        
    def get_atom_adj(self, t = 4):
        if len(self.distance_graphs) == 1:
            self.get_atom_dists()
        
        self.adjacency_graphs = (self.distance_graphs < t).astype(int)
        
        return self.adjacency_graphs
    
    def get_elec_adj(self):
        if len(self.adjacency_graphs) == 1:
            self.get_atom_adj()
            
        total_val = 0
        
        for i in range(len(self.valence_list[0])):
            total_val += self.valence_list[0][i]
        
        self.elec_adjacency_graphs = np.zeros((len(self.valence_list), total_val, total_val))
        
        curr_n, curr_m = 0, 0
        
        for i in range(len(self.adjacency_graphs)):
            for j in range(len(self.adjacency_graphs[0])):
                for b in range(self.valence_list[i][j]):
                    for k in range(len(self.adjacency_graphs[0][0])):
                        for a in range(self.valence_list[i][k]):
                            self.elec_adjacency_graphs[i][curr_n][curr_m] = self.adjacency_graphs[i][j][k]
                            curr_m += 1
                    curr_m = 0
                    curr_n += 1
            curr_n = 0
            
        return self.elec_adjacency_graphs
    
    def get_cont_ent(self):
        self.cont_ent = np.zeros(len(self.elec_adjacency_graphs), int)
        n = len(self.valence_list)
        first_val = 0
        for i in range(len(self.elec_adjacency_graphs)):
            m = 0
            for j in range(len(self.elec_adjacency_graphs[i])):
                for k in range(len(self.elec_adjacency_graphs[i, j])):
                    if k > j and self.elec_adjacency_graphs[i, j, k] > 0:
                        m += 1
            ent_val = (m/2)*(np.log(n) - 0.8378770664093455)
            if i == 0:
                first_val = ent_val
            self.cont_ent[i] = ent_val - first_val
        return self.cont_ent
    
    def make_eigenvalues(self, hamiltonian_iter=10):
        elec_count = len(self.elec_adjacency_graphs[0])
        self.eigenvalues = np.zeros((len(self.elec_adjacency_graphs), elec_count * hamiltonian_iter))
        for frame in range(len(self.elec_adjacency_graphs)):
            frame_eigs = []
            for i in range(hamiltonian_iter):
                r = np.random.normal(size=(elec_count, elec_count))
                rt = np.transpose(r)
                h = (r + rt) / np.sqrt(2 * elec_count)          
                adj_r = self.elec_adjacency_graphs[frame] * h
                eigs = np.ndarray.tolist(np.linalg.eigvals(adj_r))
                for i in range(len(eigs)):
                    frame_eigs.append(np.real(eigs[i]))
            self.eigenvalues[frame] = frame_eigs
        return self.eigenvalues
    
    def get_bin_probs(self, bin_num=1000):
        self.bin_probs = np.zeros((len(self.eigenvalues), bin_num))
        for frame in range(len(self.eigenvalues)):
            hist = np.histogram(self.eigenvalues[frame], bins=bin_num)
            prob = hist[0] / hist[0].sum()
            self.bin_probs[frame] = prob
            
    def calculate_entropy(self):
        self.entropy = np.ndarray(len(self.bin_probs))
        for frame in range(len(self.bin_probs)):
            ent = 0
            for prob in self.bin_probs[frame]:
                if prob != 0:
                    ent += prob * np.log(prob)
            self.entropy[frame] = -ent
        return self.entropy
    
    def calculate_energy(self):
        self.energy = np.ndarray(len(self.bin_probs))
        for frame in range(len(self.bin_probs)):
            en = 0
            for prob in self.bin_probs[frame]:
                if prob != 0:
                    en += np.log(prob / (1 - prob))
            self.energy[frame] = -en
        return self.energy
    
if __name__ == "__main__":
    full = Adj_Mats('full_fully.pdb')
    full.get_atom_dists()
    for i in [4, 5, 6, 7, 8, 10]:
        full.get_atom_adj(i)
        full.get_elec_adj()
        full.get_cont_ent()
        np.savetxt('full_traj_ents_{}A.csv'.format(i), full.cont_ent, delimiter=',')
    

    

        