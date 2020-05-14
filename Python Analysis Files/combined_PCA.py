#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:34:02 2019

@author: tatumhennig
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn import mixture
import matplotlib as mpl
import itertools
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal as mvn
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


##### LOADING IN DATA #########################################################

name = 'm9_allpH_h2o_c6_cossin'
data = pd.read_csv(name + '.csv')
data.set_index('Unnamed: 0',inplace=True)   # frame numbers

##### PERFORM PCA #############################################################

# taking sine and cosine of phi psi angles
datacos = data.apply(lambda x: np.cos(x*np.pi/180))
datasin = data.apply(lambda x: np.sin(x*np.pi/180))
datalinear = pd.concat([datacos,datasin],axis=1)
datalinear.to_csv( "cs-wt_all_pH_c6_phipsi.csv", index=False, encoding='utf-8-sig')


# center and scale data
    # -> changes avg value to zero
    # -> changes std dev to one
scaled_data = preprocessing.scale(data)

pca = PCA()     # creating the PCA object
pca.fit(scaled_data)    # math!
pca_data = pca.transform(scaled_data)   # PCA coordinates

# setting up labels and calculating explained variance
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# PCA!
pca_df = pd.DataFrame(pca_data, columns=labels)


##### PLOTS ###################################################################

# Scree Plot
fig = plt.figure()
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.xticks(rotation=90)
plt.title('Scree Plot - GrBP5 WT, all pH, graphene')
plt.show()
fig.savefig('scree_' + name + '.png')

# Plot using PC1 and PC2
fig2 = plt.figure()
plt.scatter(pca_df.PC1, pca_df.PC2,s=0.01)
plt.title(' Phi Psi Direct Hilbert PCA Graph - GrBP5 WT, all pH, graphene')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1])) 
plt.show()
fig2.savefig('PC1_PC2_' + name + '.png')

# Plot using PC1, PC2, and PC3
ax = plt.axes(projection = '3d')
ax.scatter3D(pca_df.PC1, pca_df.PC2, pca_df.PC3, s=0.01)
ax.set_title('PCA Graph - GrBP5 WT + M9 all pH, H2O + C6')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))
plt.savefig('3D PCA - ' + name + '.png')



# density plot in 2D
# pH 3
x = 19986
y = 39971

# pH 7
x = 39972
y = 59958

# pH 9
x = 0
y = 19985

# diff pHs
levels = np.linspace(0, 0.7, 25)
ax = sns.kdeplot(pca_df.PC1[x:y],pca_df.PC2[x:y],n_levels=levels,
                 shade=True,cmap='terrain_r', shade_lowest=False, cbar=True)
ax.set_title('Direct Hilbert PCA Graph - GrBP5 WT pH 3 graphene')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
#ax.set_xlim(-3.5, 3.5)
#ax.set_ylim(-3.5, 3.5)
plt.scatter(-0.3,-1.1)

plt.savefig('DensityPCA_wt_pH3_c6.png')

# close ups
levels = np.linspace(0, 0.7, 25)
ax = sns.kdeplot(pca_df.PC1[x:y],pca_df.PC2[x:y],n_levels=levels,
                 shade=True,cmap='terrain_r', shade_lowest=False, cbar=True)
ax.set_xlim(-2.5, 2)
ax.set_ylim(-2, 2.5)
plt.savefig('DensityPCA_wt_pH9_c6_closeup.png')

# all together now
levels = np.linspace(0, 0.25, 30)
ax = sns.kdeplot(pca_df.PC1,pca_df.PC2,n_levels=levels,
                 shade=True, cmap="terrain_r", shade_lowest=False, cbar=True)
ax.set_title('Direct Hilbert PCA Graph - GrBP5 WT all pH, graphene')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
plt.savefig('DensityPCA_wt_pH_c6.png')

# frames in the PC space
levels = np.linspace(0, 0.7, 30)
ax = sns.kdeplot(pca_df.PC1,pca_df.PC2,n_levels=levels,
                 shade=True, cmap="terrain_r", shade_lowest=False, cbar=True)
ax.set_title('Direct Hilbert PCA Graph - GrBP5 WT all pH, graphene')
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
plt.savefig('DensityPCA_wt_pH_c6.png')


# default colors
colors = ["black"]
sns.set_palette(sns.xkcd_palette(colors))
sns.palplot(sns.color_palette())
sns.palplot(sns.xkcd_palette(colors))

# getting entropy
import numpy as np

def entropy(x):
   counts = np.histogramdd(x)[0]
   dist = counts / np.sum(counts)
   logs = np.log2(np.where(dist > 0, dist, 1))
   return -np.sum(dist * logs)

x = pca_df.iloc[:,0:2].values
h = entropy(x)

def entropyln(x):
   counts = np.histogramdd(x)[0]
   dist = counts / np.sum(counts)
   logs = np.log(np.where(dist > 0, dist, 1))
   return -np.sum(dist * logs)

x = pca_df.iloc[39984:59969,0:2].values
h = entropyln(x)

# probability maps
xmin = pca_df.PC1[x:y].min()
xmax = pca_df.PC1[x:y].max()
ymin = pca_df.PC2[x:y].min()
ymax = pca_df.PC2[x:y].max()
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([pca_df.PC1[x:y], pca_df.PC2[x:y]])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
Z1 = Z*((xmax-xmin)/100)*((ymax-ymin)/100) # probability 
G = -1*1.3807e-23*np.log(Z1) # Gibbs free energy

ax = sns.heatmap(G, cmap="BuPu")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.title('Ramachandra PCA Graph - GrBP5 WT, pH 9, C6')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.savefig('prob_heatmap_pH9' + name + '.png')


# gif 3D PC frames
a = 99966
b = 119952

xy = np.vstack([pca_df.PC1[a:b],pca_df.PC2[a:b]])
z = gaussian_kde(xy)(xy)
x = 0
cmap = mpl.cm.gist_ncar_r
norm = mpl.colors.Normalize(vmin=0, vmax=0.72)
for i in range(361) :
    ax = plt.axes(projection = '3d')
    ax.scatter3D(pca_df.PC1[a:b],pca_df.PC2[a:b], z, c=z, cmap=cmap, norm=norm,s=0.01)
    ax.set_title('Hilbert PCA 3D Graph - GrBP5 WT pH 3, C6')
    ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
    ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
    ax.set_zlabel('Density')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 0.8)
    ax.view_init(azim=x)
    plt.savefig(str(i) + 'wt_pH3_c6.png')
    plt.close()
    x = x + 1
    

# find peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(z, height=0.05)
plt.plot(z)
plt.plot(peaks, z[peaks], "time")
plt.plot(np.zeros_like(z), "--", color="gray")
plt.show()
    

# gif time frames
x = 0
for i in range(400) :
    plt.scatter(pca_df.PC1, pca_df.PC2,s=0.01)
    plt.title('PCA Graph - GrBP5 WT, pH 7, H2O')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.scatter(pca_df.PC1[x], pca_df.PC2[x], color='r')
    plt.savefig('frame_' + str(i) + name + '.png')
    plt.close()
    x = x + 50
    


##### LOADING SCORES ##########################################################

# loading scores & sort by magnitude

# get names for PC1
loading_scores1 = pd.Series(pca.components_[0])
sorted_loading_scores1 = loading_scores1.abs().sort_values(ascending=False)
PC1_ls = sorted_loading_scores1[0:40].index.values
print(loading_scores1[PC1_ls])

# get names for PC2
loading_scores2 = pd.Series(pca.components_[1])
sorted_loading_scores2 = loading_scores2.abs().sort_values(ascending=False)
PC2_ls = sorted_loading_scores2[0:40].index.values
print(loading_scores2[PC2_ls])

# get names for PC3
loading_scores3 = pd.Series(pca.components_[2])
sorted_loading_scores3 = loading_scores3.abs().sort_values(ascending=False)
PC3_ls = sorted_loading_scores3[0:20].index.values
print(loading_scores3[PC3_ls])


##### GAUSSIAN MIXTURE ########################################################

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'crimson', 'g', 'darkviolet',
                              'darkgoldenrod', 'teal', 'purple', 'burlywood'])

# gaussian mixture
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=.1, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
#    plt.xlim(-5, 5)
#    plt.ylim(-5, 5)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

##### OPTIMAL CLUSTERING ######################################################

def opt_clust_dim_counts(pca_df):
    clusters = 1
    dims = 3
    curr_cov = 0
    last_cov = 1
    while np.round(curr_cov, decimals = 0) != np.round(last_cov, decimals = 0):
        if curr_cov != 0:
            last_cov = curr_cov
        gmm = mixture.GaussianMixture(n_components = clusters).fit(pca_df.iloc[:, 0:dims].values)
        curr_cov = np.sum(gmm.covariances_)
        clusters += 1
    return clusters
    
optimal_clusters = opt_clust_dim_counts(pca_df)

##### CLUSTERING ##############################################################

n=3

fig3 = plt.figure()
gmm = mixture.GaussianMixture(n_components=n).fit(pca_df.iloc[x:y,0:2].values)
predictions = gmm.predict(pca_df.iloc[x:y,0:2].values)
plot_results(pca_df.iloc[x:y,0:2].values, gmm.predict(pca_df.iloc[x:y,0:2].values), 
             gmm.means_, gmm.covariances_, 0, '')

#fig3.savefig('gauss_mix_' + name + '.png')


##### PREDICTIONS #############################################################

predictions = gmm.predict(pca_df.iloc[:,0:2].values)
fig4 = plt.figure()
plt.hist(predictions)
plt.title('Cluster Histogram - GrBP5 WT + M2 all temp, H2O')
plt.ylabel('Number of frames')
plt.xticks(())
#fig4.savefig('predict_hist_' + name + '.png')


##### MEANS ###################################################################

means=gmm.means_

#####################
x = 0
y = 19985

x = 19986
y = 39972

#levels = np.linspace(0, 1.2, 50)
ax = sns.kdeplot(pca_df.PC1[x:y],pca_df.PC2[x:y],n_levels=50,
                 shade=True,cmap='terrain_r', shade_lowest=False, cbar=True)
#ax.set_title('Direct Hilbert PCA Graph - GrBP5 WT pH 3 graphene')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))

plt.scatter(3, 3.4)

####################
time_frames1 = pca_df.index[(pca_df.PC1<3+0.02) & 
                            (pca_df.PC1>3-0.02) & 
                            (pca_df.PC2<3.4+0.02) & 
                            (pca_df.PC2>3.4-0.02)]


##### SIMULATION PERCENTAGES ##################################################

cfs = {}

for l in range(n):
    cluster = []
    for j in range(len(predictions)):
        if predictions[j] == l:
            cluster.append(j)
            
    one = 0
    for i in cluster:
        if i <= 19997 :
            one += 1       
    two = 0
    for i in cluster:
        if i > 19997 and i <= 39983 :
            two += 1   
    three = 0
    for i in cluster:
        if i >39983 and i <=59969 :
            three += 1      
    four = 0
    for i in cluster:
        if i >59969 :
            four += 1

    cc = [one, two, three, four]
    cf = []
    for i in range(len(cc)):
        cf.append((cc[i]/len(cluster))*100)
    
    cfs[l] = cf
    
cf_df = pd.DataFrame.from_dict(cfs)
            
##### FOR CLUSTER RAMACHANDRANS ###############################################

#data['cluster_num']=predictions
d = data.iloc[59970:79968]

##### SIMULATION CLUSTERING ###################################################

fig5 = plt.figure()
plt.scatter(pca_df.PC1[0:19985], pca_df.PC2[0:19985],s=0.01, color='purple')
plt.scatter(pca_df.PC1[19986:39971], pca_df.PC2[19986:39971], s=0.01, color='green')
plt.scatter(pca_df.PC1[39972:59958], pca_df.PC2[39972:59958], s=0.01, color='orange')
plt.title('Direct Hilbert PCA Graph - GrBP5 WT all pH, graphene')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1])) 
fig5.savefig('sim_clust_' + name + '.png')


