"""
This module ingests full trajectory csvs describing the energy and entropy trajectories and visualizes them

Author: David Corbo

Last Edited: 2/24/20
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kde

class Traj(object):
    
    def __init__(self, name, time=100):
        self.name = name
        if not os.path.exists("{}_entropy.csv".format(name)) or not os.path.exists("{}_energy.csv".format(name)) or not os.path.exists("{}_entropy_ref.csv".format(name)) or not os.path.exists("{}_energy_ref.csv".format(name)):
            raise OSError("There are one or more missing files.")
        else:
            self.ent = np.genfromtxt("{}_entropy.csv".format(name), delimiter=',')
            self.eng = np.genfromtxt("{}_energy.csv".format(name), delimiter=',')
            self.ent_ref = [np.genfromtxt("{}_entropy_ref.csv".format(name), delimiter=',')[1]] * len(self.ent) 
            self.eng_ref = [np.genfromtxt("{}_energy_ref.csv".format(name), delimiter=',')[1]] * len(self.eng)
            self.frames = np.arange(0, len(self.ent))
            self.frames_time = time * (self.frames / len(self.frames))
        
    def make_ent_eng(self, i):
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.frames_time, self.ent, lw=.15, c='blue')
        plt.plot(self.frames_time, self.ent_ref, lw=.7, c='red')
        plt.xlabel("Time (ns)")
        plt.ylabel("Entropy (J/K)")
        plt.title("Entropy vs. Time {}A".format(i))
        plt.show()
        fig.savefig("{}_entropy.png".format(self.name), dpi=200)
        plt.clf()
        
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.frames_time, self.eng, lw=.15, c='green')
        plt.plot(self.frames_time, self.eng_ref, lw=.7, c='red')
        plt.xlabel("Time (ns)")
        plt.ylabel("Energy (J)")
        plt.title("Energy vs. Time {}A".format(i))
        plt.show()
        fig.savefig("{}_energy.png".format(self.name), dpi=200)
        plt.clf()
    
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(self.eng, self.ent, s=1.5, c='blue')
        plt.scatter(self.eng_ref, self.ent_ref, s=16, c='red')
        plt.xlabel("Energy (J)")
        plt.ylabel("Entropy (J/K)")
        plt.title("Entropy vs Energy {}A".format(i))
        plt.show()
        fig.savefig("{}_combined.png".format(self.name), dpi=200)
        plt.clf()
        
        fig = plt.figure(figsize=(12, 8))
        nbins = 300
        k = kde.gaussian_kde([self.eng, self.ent])
        xi, yi = np.mgrid[self.eng.min():self.eng.max():nbins*1j, self.ent.min():self.ent.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
        plt.xlabel("Energy (J)")
        plt.ylabel("Entropy (J/K)")
        plt.title("Entropy vs Energy {}A".format(i))
        plt.show()
        fig.savefig("{}_combined_dens.png".format(self.name), dpi=200)
        plt.clf()
        
if __name__ == "__main__":
    for i in [4, 5, 6, 7, 8, 10]:
        temp = Traj("LK7_{}A".format(i), time=100)
        temp.make_ent_eng(i)
        
    
        