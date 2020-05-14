import numpy as np
import os
import sys

class Adj_Mats():
    def __init__(self, pdb):
        self.file = pdb
        self.d_graphs = np.zeros(1, int)
        self.a_graphs = np.zeros(1, int)
    
    def set_AtomDists(self, new_dists):
        self.d_graphs = new_dists
        
    def set_AtomAdj(self, new_adj):
        self.a_graphs = new_adj
    
    #Adapted from: xplo2xyz.py on Open NMR Project (http://nmrwiki.org/wiki/index.php?title=Script_xplo2xyz.py_-_convert_PDB_files_created_by_NIH-XPLOR_to_XYZ_file_format)
    def get_AtomDists(self):
        class PDBAtom(object):
            def __init__(self, string):
                #this is what we need to parse
                #ATOM      1  CA  ORN     1       4.935   1.171   7.983  1.00  0.00      sega
                #XPLOR pdb files do not fully agree with the PDB conventions 
                self.name = string[12:16].strip()
                self.x = float(string[30:38].strip())
                self.y = float(string[38:46].strip())
                self.z = float(string[46:54].strip())
                self.warnings = []
                if len(string) < 78:
                    self.element = self.name[0]
                    self.warnings.append('Chemical element name guessed ' +\
                                         'to be %s from atom name %s' % (self.element, self.name))
                else:
                    self.element = string[76:78].strip()
                    
        if os.path.isfile(self.file):
            pdb_file = open(self.file,'r')
        else:
            raise Exception('file %s does not exist' % self.file)

        lineno = 0
        frames = []
        atoms = []
        #read pdb file
        for line in pdb_file:
            lineno += 1
            if line.startswith('ATOM'):
                try:
                    at_obj = PDBAtom(line)
                    atoms.append([at_obj.x, at_obj.y, at_obj.z])
                except:
                    sys.stderr.write('\nProblem parsing line %d in file %s\n' % (lineno,self.file))
                    sys.stderr.write(line)
                    sys.stderr.write('Probably ATOM entry is formatted incorrectly?\n')
                    sys.stderr.write('Please refer to - http://www.wwpdb.org/documentation/format32/sect9.html#ATOM\n\n')
                    sys.exit(1)
            elif line.startswith('END'):
                frames.append(atoms)
                atoms = []
        pdb_file.close()

#       NUMPY METHOD
        base = np.zeros((len(frames), len(frames[0]), 3))
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                for k in range(len(frames[i][j])):
                    base[i][j][k] = frames[i][j][k]
        dists = np.reshape(base, (len(frames), 1, len(frames[0]), 3)) - np.reshape(base, (len(frames), len(frames[0]), 1, 3))
        dists = dists**2
        dists = dists.sum(3)
        dists = np.sqrt(dists)
        self.d_graphs = dists
        
#        PANDAS METHOD (slow)
#        for frame in frames:
#            graph = pd.DataFrame(index = range(len(frame)), columns = range(len(frame)))
#            for i, atom1 in enumerate(frame):
#                for j, atom2 in enumerate(frame):
#                    if i == j:
#                        graph.iloc[i, j] = 0
#                    if i < j:
#                        graph.iloc[i, j] = np.sqrt((atom1.x - atom2.x)**2 +
#                                  (atom1.y - atom2.y)**2 + (atom1.z - atom2.z)**2)
#                        graph.iloc[j, i] = graph.iloc[i, j]
#            self.d_graphs.append(graph)
        
        return self.d_graphs
    
    #Parameter:
    #   -t: The threshold distance for adjacency in Angstroms (4-25)
    def get_AtomAdj(self, t = 4):
        # NUMPY METHOD
        if len(self.d_graphs) == 1:
            self.get_AtomDists()
        
        self.a_graphs = (self.d_graphs < t).astype(int)
        
        # PANDAS METHOD (slow)
        # self.a_graphs = []
        # if not self.d_graphs:
        #     self.get_AtomDists()
        
        # for graph in self.d_graphs:
        #     adj = graph.le(t).stack()
        #     adj_pos = adj[adj].index.values
        #     adj_graph = pd.DataFrame(index = graph.index, columns = graph.columns)
        #     for pos in adj_pos:
        #         adj_graph.iloc[pos[0], pos[1]] = 1
        #     adj_graph.fillna(0)
        #     self.a_graphs.append(adj_graph)        
        
        return self.a_graphs
