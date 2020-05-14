import Bio.PDB as bpdb
import numpy as np
import pandas as pd
import easygui as eg
import multiprocessing.dummy as mp

#from time import time
parser = bpdb.PDBParser()
file = eg.fileopenbox(filetypes = ['*.pdb'])
structure = parser.get_structure('4A_s2', file)

angles_by_frame = pd.DataFrame(columns = np.linspace(1,4,num = 4))

frame = 1
clmns = []
rows={}
for i in range(2):
    clmns.append('phi' f'{i+2}')
    clmns.append('psi' f'{i+2}')

model_list = bpdb.Selection.unfold_entities(structure, 'M')
with mp.Pool(32) as pool:    
    chain_list = pool.map(lambda x: x['A'], model_list)
    poly_list = pool.map(lambda x: bpdb.Polypeptide.Polypeptide(x), chain_list)
    angle_list = pool.map(lambda x: x.get_phi_psi_list(), poly_list)
    rowstuff = pool.map(lambda x: np.reshape(x,[1,len(x)*2])[0][2:-2] * (180/np.pi), angle_list)
    rowlist = list(rowstuff)


angles_by_frame = pd.DataFrame(rowlist,index=np.linspace(1,len(rowlist),num=len(rowlist)),columns=clmns)
angles_by_frame.to_csv('/Users/tatumhennig/Desktop/4A_500ns_sim2.csv')


