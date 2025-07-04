from datetime import datetime
startTime = datetime.now()
from skimage.segmentation import watershed
from numpy import inf
import matplotlib
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report

from scipy.stats import wasserstein_distance
from scipy.stats import wasserstein_distance_nd
from scipy.special import rel_entr
from scipy.special import kl_div
from scipy.spatial import distance

import seaborn as sn

import random

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import pandas as pd
from itertools import chain, repeat, compress, islice
import itertools
from numpy import asarray
from numpy import savetxt
import os
import math
from functools import reduce
from numpy import linalg as la
import pdb
import matplotlib.pyplot as plt
from collections import Counter
import json

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import plotly.express as px

from mpl_toolkits.axes_grid1 import make_axes_locatable


import plotly.graph_objects as go


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


from sklearn.cluster import KMeans
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat)
import skimage.segmentation as seg
from skimage.filters import threshold_multiotsu
from skimage.filters import try_all_threshold
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)


from PIL import Image
import matplotlib.image as mpimg

from skimage import measure
from skimage.measure import label, regionprops
from skimage import filters

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from skimage import exposure
from sklearn.ensemble import GradientBoostingClassifier

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from skimage import data, segmentation, feature, future
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from functools import partial

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


import sys

from sklearn.model_selection import train_test_split

from scipy.stats import entropy

from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from matplotlib.colors import LightSource
from matplotlib import cbook, cm

from skimage import data, segmentation, feature, future
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from functools import partial




#learn = XGBClassifier()
#learn.fit(X_train, y_train)

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim



def n_co2_ads_des(num_a,atom_type2,atom_type2_z, cutoff1, cutoff2):
    a2_keep_x=[]                    
    a2_keep_y=[]
    a2_keep_z=[]
    n2_keep=[]
    a2_des=[]                    
    n2_des=[]

    for na2,atom2,atom2_z in zip(num_a,atom_type2,atom_type2_z):
        z2_keep_x=[]
        z2_keep_y=[]
        z2_keep_z=[]
        zn2_keep=[]
        z2_des=[]
        zn2_des=[]

        for n2,a2,a2_z in zip(na2,atom2,atom2_z):
            if a2_z < cutoff2 or a2_z > cutoff1: 
               z2_keep_x.append(a2[:1])
               z2_keep_y.append(a2[1:2])
               z2_keep_z.append(a2[2:3])

               zeta2_arr_x=np.array(z2_keep_x)	   
               zeta2_arr_y=np.array(z2_keep_y)	   
               zeta2_arr_z=np.array(z2_keep_z)

               zn2_keep.append(n2)
               n2_arr=np.array(zn2_keep)	          


               #n2r = np.array(n2).reshape(1,1)
               #a2r = np.array(a2).reshape(1,3)
               #print(n2r.shape)
               #print(a2r.shape)
               #an2r = np.concatenate((n2r,a2r), axis = 1)
               #zn2_keep.append(an2r)
               #n2_arr=np.array(zn2_keep)	 


            else:
               z2_des.append(a2)
               zeta2_arr_d=np.array(z2_des)	   
               zn2_des.append(n2)
               n2_arr_d=np.array(zn2_des)	    

               #n2r = np.array(n2).reshape(1,1)
               #a2r = np.array(a2).reshape(1,3)
               #an2r = np.concatenate((n2r,a2r), axis = 1)
               #zn2_des.append(an2r)
               #n2_arr_d=np.array(zn2_des)	 




        a2_keep_x.append(zeta2_arr_x)
        a2_keep_y.append(zeta2_arr_y)
        a2_keep_z.append(zeta2_arr_z)

        n2_keep.append(n2_arr)

        a2_des.append(zeta2_arr_d)
        n2_des.append(n2_arr_d)



    atom2_z_keep_x=np.array(a2_keep_x,dtype=object)	   
    atom2_z_keep_y=np.array(a2_keep_y,dtype=object)	   
    atom2_z_keep_z=np.array(a2_keep_z,dtype=object)	   



    num_a2_z_keep=np.array(n2_keep,dtype=object)	    


    atom2_z_des=np.array(a2_des,dtype=object)	   
    num_a2_z_des=np.array(n2_des,dtype=object)	    

    return(num_a2_z_keep,atom2_z_keep_x,atom2_z_keep_y,atom2_z_keep_z,num_a2_z_des,atom2_z_des)




def n_co2_ads(num_a,atom_type2,atom_type2_z, cutoff1, cutoff2):
    a2_keep=[]                    
    n2_keep=[]
    for na2,atom2,atom2_z in zip(num_a,atom_type2,atom_type2_z):
        z2_keep=[]
        zn2_keep=[]
        for n2,a2,a2_z in zip(na2,atom2,atom2_z):
            if a2_z < cutoff2 or a2_z > cutoff1: 
               z2_keep.append(a2)
               zeta2_arr=np.array(z2_keep)	   
               zn2_keep.append(n2)
               n2_arr=np.array(zn2_keep)	          
        a2_keep.append(zeta2_arr)
        n2_keep.append(n2_arr)
    atom2_z_keep=np.array(a2_keep,dtype=object)	   
    num_a2_z_keep=np.array(n2_keep,dtype=object)	    
    return(num_a2_z_keep,atom2_z_keep)


def windowing(coord_splitted):

    inter = []
    start =   10 #  10   #   10   # 10     # 10      #10     # 10   # 0
    fin   =  250 #  250  #  250   # 80     # 230     #30     # 410  # 200
    end   =  730 #  9610 #  730   # 20     # 9690   #970    # 9530 # 300
    final =  970 #  9850 #  970   # 90     # 9910   #990    # 9930 # 500
    step  =  240 #  240  #  240   # 2  # 5 # 220   #2      # 40   # 20
    
    for fr in range(end):
        inter.append(np.array(coord_splitted[start:fin])) 
        start = start + step 
        fin   = fin   + step 
        if start == end:
           break   
        inter_arr = np.array(inter)

    return(inter_arr)

def coor_all_multi_bis(intervals,xyzs):

    all_count_abs = []
    all_count_c   = []
    all_count_p   = []
    all_count     = []

    for o, (coord_splitted,xyz) in enumerate(zip(intervals,xyzs)):

        decay = []
        decay_x = []
        maskns = []
        m = 0
        count = []
        count_c = []
        count_p = []
        splitn = coord_splitted[0][:]  #  [:,0]
        split_xyz = xyz[0][:]
        x_sp = split_xyz[:,0].reshape(len(split_xyz[:,0]),1)

        for n, (i,j) in enumerate(zip(coord_splitted,xyz)):
             
            #j = j[:,0].reshape(len(j),1)
            split0 = coord_splitted[0][:] #  [:,0]  #.reshape(len(coord_splitted[0][:,1]),1) 
            splitm = coord_splitted[m][:] #  [:,0]  #.reshape(len(coord_splitted[m][:,1]),1)
            #print(split0.shape, splitm.shape, splitn.shape)
            mask0 = np.isin(splitn, split0)     
            split2 = splitn[mask0]

            #print(mask0.size)

            x_sp2 = x_sp[mask0]
            maskn = np.isin(split2, splitm)     

            #print(maskn.size)

            split3 = split2[maskn]
            x_sp3 = x_sp2[maskn]

            #print(split3.shape)
            decay.append(split3)
            count.append(len(split3))
            maskns.append(maskn)
            decay_x.append(x_sp3)

            splitn = split3
            x_sp = x_sp3



            if n > 0:
               m = n
        
        count = count[1:]
        ext   = [count[-1]]
        count.extend(ext)
        count_abs = count
        count = np.array(count)/count[0]

        decay = decay[1:]
        ext1  = [decay[-1]]
        decay.extend(ext1)

        decay_x = decay_x[1:]
        ext2  = [decay_x[-1]]
        decay_x.extend(ext2)


        all_count_abs.append(count_abs)
        all_count.append(count)
        all_count_c.append(decay)
        all_count_p.append(decay_x)


    all_count_abs  = np.array(all_count_abs) 
    all_count      = np.array(all_count)
    all_count_c    = np.array(all_count_c)
    all_count_p    = np.array(all_count_p)

    return(all_count_abs,all_count,all_count_c,all_count_p)



def read_rdf(filename):
    rdf=[]
    with open(filename) as f:
         lines=f.readlines()
         for line in lines:
             rdf.append(line.strip().split("\n"))
         rdf=np.array(rdf).reshape(len(rdf))    
         rdf = rdf.astype(float)	 
         return(rdf)    

def con_csv_bis(df):
    co2_x = []
    for row in df["0"]:
        row = row.replace(')','')
        row = row.replace(',','')
        row = row.replace('array(','')
        row = row.replace('\n','') 
        row = row.replace("[",'') 
        row = row.replace("]",'') 
        row = row.replace("     ",' ') 
        row = row.replace("    ",' ') 
        row = row.replace("   ",' ') 
        row = row.replace("  ",' ') 

        row = row.strip().split(' ') # , '  ')  # .split('  ') 
        arr = np.array(row)
        arr = arr.astype(float)
        co2_x.append(arr)

    return(co2_x)

# Check for high density regions at the edge 

def edges_pbc(coor,k):

    cor_xinf = []
    cor_xsup = []
    num_inf = []
    num_sup = []
    
    for n,cor in enumerate(coor,k):
        n = n + 1 
        #print(cor.shape)
        x_inf = np.where(cor[:,k] < 10)
        xinf = cor[x_inf]
    
        x_sup = np.where(cor[:,k] > 490)
        xsup = cor[x_sup]
        
        if len(xinf) > 0: 
           cor_xinf.append(xinf)
           num_inf.append(n)
    
        if len(xsup) > 0: 
           cor_xsup.append(xsup)
           num_sup.append(n)
    #print(n) 
    return(cor_xinf,cor_xsup,num_inf,num_sup)

# Check for high density regions connected over PBC 

def conn_pbc(cor_inf,cor_sup,num_inf,num_sup,k): 
    pbi = []
    pbj = []

    for i,ni in zip(cor_inf,num_inf):
        for j,nj in zip(cor_sup,num_sup):

            ik = np.isin(i[:,k],j[:,k])
            jk = np.isin(j[:,k],i[:,k])

            ki = i[ik]
            kj = j[jk]

            if len(ki) > 0:
               pbi.append(ni)

            if len(kj) > 0:
               pbj.append(nj)

    pbi_ar = np.array(pbi)
    pbj_ar = np.array(pbj)

    pbi_ar = pbi_ar.reshape(len(pbi_ar),1)
    pbj_ar = pbj_ar.reshape(len(pbi_ar),1)

    pb = np.concatenate((pbi_ar,pbj_ar),axis = 1)

    return(pb)

def boundary(regio,size):
    coordinates = []
    n_regions = []
    coordinates_off = []
    n_regions_off = []
    areas = [] 
    nx_sh1 , ny_sh1 = 500 , 500
    coor_flat = []
    coor_flat_off = []
    for region in regionprops(regio):
        #print("reg",region.area)
        if region.area >= size:
           are = region.area
           reg = region.coords
           reg1= reg + 1 
           coordinates.append(np.array(reg))
           n_regions.append(region)
           areas.append(np.array(are))
           coor_flat.append(np.array(reg1))

        if region.area < size:
           reg = region.coords
           reg1= reg + 1 
           coordinates_off.append(np.array(reg))
           n_regions_off.append(region)
           coor_flat_off.append(np.array(reg1))

    coor_flat = np.array(coor_flat,dtype=object )
    return(coordinates,n_regions,coordinates_off,n_regions_off,areas,coor_flat,coor_flat_off)

# Check for overlapping of regions between X and Y edges 
def overlap(pb_x,pb_y):
    keep = []
    for x in pb_x:
        ind = np.isin(x,pb_y) 
        x_keep = x[ind]
        if x_keep.shape[0] > 0:
           keep.append(x_keep[0])
    keep = np.unique(keep)
    return(keep)

# Merging regions checking that when crossing both  X and Y edges the regions are merged in the same one  

def merge_reg_pbc(keep,pb):

    merge = []
    merged = []
    
    for b in pb:
        if len(keep) > 0:
           for j in keep:
               j = j.astype(int)
               if j in b:
                  ins = np.where(b == j)
                  ins_in = np.where(b != j)
                  if ins[0].shape == 1:
                     med = b[ins[0]]
                     me  = b[ins_in[0]]
                     merged.append(med[0])
                     merge.append(me[0])
        else:
           merge.append(b[1])
           merged.append(b[0])
    return(merge,merged)

# Regions re-lebelling after merging  
def relabel(merge,merged):
    for isl, isl_mer in zip(merge,merged):
        ind = np.where((highs_mer == isl))
        highs_mer[ind] = isl_mer
    return(highs_mer)    




################## CV2 clustering

inf_dfx10 =     pd.read_csv("../0_to_10ns/inf_st_co2_x_10ns.csv")
inf_dfy10 =     pd.read_csv("../0_to_10ns/inf_st_co2_y_10ns.csv")
inf_dfx25 =    pd.read_csv("../10_to_25ns/inf_st_co2_x_25ns.csv")
inf_dfy25 =    pd.read_csv("../10_to_25ns/inf_st_co2_y_25ns.csv")
inf_dfx40 =    pd.read_csv("../25_to_40ns/inf_st_co2_x_40ns.csv")
inf_dfy40 =    pd.read_csv("../25_to_40ns/inf_st_co2_y_40ns.csv")
inf_dfx55 =    pd.read_csv("../40_to_55ns/inf_st_co2_x_55ns.csv")
inf_dfy55 =    pd.read_csv("../40_to_55ns/inf_st_co2_y_55ns.csv")
inf_dfx70 =    pd.read_csv("../55_to_70ns/inf_st_co2_x_70ns.csv")
inf_dfy70 =    pd.read_csv("../55_to_70ns/inf_st_co2_y_70ns.csv")
inf_dfx85 =    pd.read_csv("../70_to_85ns/inf_st_co2_x_85ns.csv")
inf_dfy85 =    pd.read_csv("../70_to_85ns/inf_st_co2_y_85ns.csv")
inf_dfx100 =  pd.read_csv("../85_to_100ns/inf_st_co2_x_100ns.csv")
inf_dfy100 =  pd.read_csv("../85_to_100ns/inf_st_co2_y_100ns.csv")


inf_x10  = np.hstack(con_csv_bis(inf_dfx10))
inf_y10  = np.hstack(con_csv_bis(inf_dfy10))
inf_x25  = np.hstack(con_csv_bis(inf_dfx25))
inf_y25  = np.hstack(con_csv_bis(inf_dfy25))
inf_x40  = np.hstack(con_csv_bis(inf_dfx40))
inf_y40  = np.hstack(con_csv_bis(inf_dfy40))
inf_x55  = np.hstack(con_csv_bis(inf_dfx55))
inf_y55  = np.hstack(con_csv_bis(inf_dfy55))
inf_x70  = np.hstack(con_csv_bis(inf_dfx70))
inf_y70  = np.hstack(con_csv_bis(inf_dfy70))
inf_x85  = np.hstack(con_csv_bis(inf_dfx85))
inf_y85  = np.hstack(con_csv_bis(inf_dfy85))
inf_x100 = np.hstack(con_csv_bis(inf_dfx100))
inf_y100 = np.hstack(con_csv_bis(inf_dfy100))

inf_x100ns = np.concatenate((inf_x10,inf_x25,inf_x40,inf_x55,inf_x70,inf_x85,inf_x100), axis = 0)
inf_y100ns = np.concatenate((inf_y10,inf_y25,inf_y40,inf_y55,inf_y70,inf_y85,inf_y100), axis = 0)

inf_x100ns = np.hstack(inf_x100ns)
inf_y100ns = np.hstack(inf_y100ns)

nx_sh , ny_sh = 500 , 500
nx_sh1 , ny_sh1 = 501 , 501

minx = np.min(inf_x100ns) #  + 0.5
miny = np.min(inf_y100ns) #  + 0.5
maxx = np.max(inf_x100ns) #  - 0.5
maxy = np.max(inf_y100ns) #  - 0.5

xbins = np.linspace(minx,maxx,nx_sh1)
ybins = np.linspace(miny,maxy,ny_sh1)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
ret = stats.binned_statistic_2d(inf_x100ns, inf_y100ns, None, 'count', bins=[xbins, ybins],expand_binnumbers=True)

image   = np.float64(ret[0])

# Equalization

img_equ = exposure.adjust_log(image, 1)
#img_equ = np.log(image)

img_eq   = denoise_tv_chambolle(img_equ, weight=0.8)  # 0.7  0.5 0.4


np.set_printoptions(threshold=sys.maxsize)



sigma_min = 1 #1 #  1
sigma_max = 8 # 4 #  16 #  16

features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max) 

features = features_func(img_eq)
 

k_sig1 =  np.array([0])
k_sig2 =  np.array([1])
k_sig3 =  np.array([2,6,10,14])
k_sig4 =  np.array([3,7,11,15])

comb_array_sig = np.array(np.meshgrid(k_sig3, k_sig4)).T.reshape(-1, 2) 

importancess = []
rank_feat_imps = []
entrops_em = []
entrops_ne = []
entrops_hi = []
entrop_import = []

tr_entrops_em = []
tr_entrops_ne = []
tr_entrops_hi = []
tr_entrops = []

entrops = []
entrops_pred = []
wasser_d = []
nc = 16
rel_entros = []
for n,k in enumerate(comb_array_sig[:,:]):
    zeros  = np.zeros(features[:,:,0].shape)
    zeros[(features[:,:,k[1]] >  -0.005)  & (features[:,:,k[0]] > -0.005)] = 1 
    zeros[(features[:,:,k[1]] <= -0.005)  & (features[:,:,k[0]] > -0.005)] = 2 
    zeros[(features[:,:,k[1]] >= -0.005)  & (features[:,:,k[0]] < -0.005)] = 2 
    zeros[(features[:,:,k[1]] <= -0.005)  & (features[:,:,k[0]] < -0.005)] = 3 


    annotation = zeros.astype(int)
 
    sele1 = features[:,:,[k[0],k[1]]][(zeros == 1),:]
    sele_ann1 = annotation[np.where(zeros == 1)]
    
    sele2 = features[:,:,[k[0],k[1]]][(zeros == 2),:]
    sele_ann2 = annotation[np.where(zeros == 2)]
    
    sele3 = features[:,:,[k[0],k[1]]][(zeros == 3),:]
    sele_ann3 = annotation[np.where(zeros == 3)]
    
    s = [len(sele1),len(sele2),len(sele3)]
    s1 = len(sele1)
    s2 = len(sele2)
    s3 = len(sele3)

   
    zeros = zeros.astype(int)
    fig, ax1 = plt.subplots(1,1,figsize=(10,10))
    colors = ['blue','deepskyblue','red']
    ax1.imshow(zeros, cmap = "jet")

    plt.savefig('segmented_map.'+str(k[0])+'.'+str(k[1])+'.png') 

    stot = s1+s2+s3
    print(s1,s2,s3,stot)
    result = annotation - 1

    ##### Get coordinates for residence time and check for PBC


    reshaped_labels_sigma05 = result  
    reshaped_labels_sigma05_empty = np.copy(reshaped_labels_sigma05)
    reshaped_labels_sigma05_netwo = np.copy(reshaped_labels_sigma05)
    reshaped_labels_sigma05_highs = np.copy(reshaped_labels_sigma05)
    reshaped_labels_sigma05_empty[reshaped_labels_sigma05_empty != 0] = 3
    reshaped_labels_sigma05_netwo[reshaped_labels_sigma05_netwo != 1] = 3
    reshaped_labels_sigma05_highs[reshaped_labels_sigma05_highs != 2] = 3
    
    
    conn_labels_sigma05_empty, nlab_sigma05_empty = measure.label(reshaped_labels_sigma05_empty, background = 3,return_num =  True)
    conn_labels_sigma05_netwo, nlab_sigma05_netwo = measure.label(reshaped_labels_sigma05_netwo, background = 3,return_num =  True)
    conn_labels_sigma05_highs, nlab_sigma05_highs = measure.label(reshaped_labels_sigma05_highs, background = 3,return_num =  True)
    
    # Relabel network as a continous (only label 1)
    
    conn_labels_sigma05_netwo_emp  = np.copy(conn_labels_sigma05_highs)
    conn_labels_sigma05_highs_all  = np.copy(conn_labels_sigma05_highs)
    
    
    conn_labels_sigma05_highs_all[(conn_labels_sigma05_highs_all != 0)] = 1
    
    
    conn_labels_sigma05_netwo_emp[(conn_labels_sigma05_netwo_emp != 0)] = 2
    conn_labels_sigma05_netwo_emp[(conn_labels_sigma05_netwo_emp == 0)] = 1
    conn_labels_sigma05_netwo_emp[(conn_labels_sigma05_netwo_emp == 2)] = 0
    
    
    ones = np.ones(image.shape)
    ones = ones.astype(int)
    
    coordinates_highs = boundary(conn_labels_sigma05_highs,0)[0]
    num_regions_highs = boundary(conn_labels_sigma05_highs,0)[1]
    coordinates_netwo = boundary(conn_labels_sigma05_netwo,0)[5]
    num_regions_netwo = boundary(conn_labels_sigma05_netwo,0)[1]
    coordinates_empty = boundary(conn_labels_sigma05_empty,0)[5]
    num_regions_empty = boundary(conn_labels_sigma05_empty,0)[1]
    area_empty = boundary(conn_labels_sigma05_empty,0)[4]
    area_netwo = boundary(conn_labels_sigma05_netwo,0)[4]
    
    coordinates_netwo = np.vstack(coordinates_netwo)
    coordinates_empty = np.vstack(coordinates_empty)
    
    print(coordinates_netwo.shape)
    print(coordinates_empty.shape)
    
    
    np.savetxt('inf_area_empty'+str(k[0])+'.'+str(k[1])+'.txt' ,area_empty , fmt = "%s"  )
    np.savetxt('inf_area_netwo'+str(k[0])+'.'+str(k[1])+'.txt' ,area_netwo , fmt = "%s"  )
    #np.savetxt('inf_num_regions_empty'+str(k[0])+'.'+str(k[1])+'.txt' ,np.vstack(num_regions_empty)     , fmt = "%s"  )
    
    coordinates_netwo_emp = boundary(conn_labels_sigma05_netwo_emp,0)[0]
    num_regions_netwo_emp = boundary(conn_labels_sigma05_netwo_emp,0)[1]
    
    
    coordinates_highs_all = boundary(conn_labels_sigma05_highs_all,0)[0]
    num_regions_highs_all = boundary(conn_labels_sigma05_highs_all,0)[1]
    
    
    labels_all = np.copy(conn_labels_sigma05_highs)
    labels_all[(labels_all == 0)] = 1
    labels_all[(labels_all != 0)] = 1
    
    coordinates_all = boundary(labels_all,0)[0]
    
    cor_xinf  = edges_pbc(coordinates_highs,0)[0]
    cor_xsup  = edges_pbc(coordinates_highs,0)[1]
    num_xinf  = edges_pbc(coordinates_highs,0)[2]
    num_xsup  = edges_pbc(coordinates_highs,0)[3]
    
    cor_yinf  = edges_pbc(coordinates_highs,1)[0]
    cor_ysup  = edges_pbc(coordinates_highs,1)[1]
    num_yinf  = edges_pbc(coordinates_highs,1)[2]
    num_ysup  = edges_pbc(coordinates_highs,1)[3]
    
    
    pb_x = conn_pbc(cor_xinf,cor_xsup,num_xinf,num_xsup,1) # 1
    pb_y = conn_pbc(cor_yinf,cor_ysup,num_yinf,num_ysup,0) # 0
    pb = np.concatenate((pb_x,pb_y), axis = 0)
    keep = overlap(pb_x,pb_y)
    
    merge, merged = merge_reg_pbc(keep,pb)
    highs_mer = np.copy(conn_labels_sigma05_highs)
    
    highs_merged = relabel(merge,merged)
    
    #np.savetxt('inf_merged'+str(k[0])+'.'+str(k[1])+'.txt'  ,highs_merged  , fmt = "%s"  )
    
    num_regions_highs_merged     = boundary(highs_merged,0)[1]  # 25 75 25  50  100
    areas_highs_merged           = boundary(highs_merged,0)[4]  # 25 75 25  50  100
    coordinates_highs_merged     = boundary(highs_merged,0)[5]
    
    
    conn_labels_sigma05_netwo, nlab_sigma05_netwo = measure.label(conn_labels_sigma05_netwo, background = 0,return_num =  True)
    
    #np.savetxt('inf_num_regions_highs_merged'+str(k[0])+'.'+str(k[1])+'.txt'  ,np.vstack(num_regions_highs_merged)     , fmt = "%s"  )
    np.savetxt('inf_area_regions_highs_merged'+str(k[0])+'.'+str(k[1])+'.txt' ,np.vstack(areas_highs_merged)     , fmt = "%s"  )
    
    
    len_net     = len(coordinates_netwo_emp)*2
    len_net_emp = len(coordinates_netwo)*2
    
    st_coordinates_netwo_emp = np.hstack(coordinates_netwo_emp)
    st_coordinates_netwo     = np.hstack(coordinates_netwo)
    st_coordinates_empty     = np.hstack(coordinates_empty)
    
    
    len_net_emp     = st_coordinates_netwo_emp.shape[0]*2
    len_net         = st_coordinates_netwo.shape[0] # *2
    len_emp         = st_coordinates_empty.shape[0] # *2
    
    st_coordinates_netwo_emp = st_coordinates_netwo_emp.reshape(len_net_emp)
    st_coordinates_netwo     =     st_coordinates_netwo.reshape(len_net)    
    st_coordinates_empty     =     st_coordinates_empty.reshape(len_emp)    
    
    
    np.savetxt('inf_coordinates_netwo_emp'+str(k[0])+'.'+str(k[1])+'.txt' ,st_coordinates_netwo_emp  , fmt = "%s"  )
    

    tag = np.array([k[0],k[1]])

    conn_labels_sigma05_netwo_emp = pd.DataFrame(conn_labels_sigma05_netwo_emp)
    #conn_labels_sigma05_netwo_emp.to_csv('inf_conn_labels_netwo_emp_{}.csv'.format(tag))  
    
    
    highs = pd.DataFrame(coordinates_highs_merged)
    highs.to_csv('inf_coor_high_{}.csv'.format(tag))  
    
    highs = np.zeros(image.shape)
    
    highs_conn, nlab_highs_conn = measure.label(highs, background = 0,return_num =  True)
    #np.savetxt('inf_conn_highs'+str(k[0])+'.'+str(k[1])+'.txt' ,highs_conn , fmt = "%s"  )



#################################################################################33
print("END INF, START SUP")
################## CV2 clustering

sup_dfx10 =     pd.read_csv("../0_to_10ns/sup_st_co2_x_10ns.csv")
sup_dfy10 =     pd.read_csv("../0_to_10ns/sup_st_co2_y_10ns.csv")
sup_dfx25 =    pd.read_csv("../10_to_25ns/sup_st_co2_x_25ns.csv")
sup_dfy25 =    pd.read_csv("../10_to_25ns/sup_st_co2_y_25ns.csv")
sup_dfx40 =    pd.read_csv("../25_to_40ns/sup_st_co2_x_40ns.csv")
sup_dfy40 =    pd.read_csv("../25_to_40ns/sup_st_co2_y_40ns.csv")
sup_dfx55 =    pd.read_csv("../40_to_55ns/sup_st_co2_x_55ns.csv")
sup_dfy55 =    pd.read_csv("../40_to_55ns/sup_st_co2_y_55ns.csv")
sup_dfx70 =    pd.read_csv("../55_to_70ns/sup_st_co2_x_70ns.csv")
sup_dfy70 =    pd.read_csv("../55_to_70ns/sup_st_co2_y_70ns.csv")
sup_dfx85 =    pd.read_csv("../70_to_85ns/sup_st_co2_x_85ns.csv")
sup_dfy85 =    pd.read_csv("../70_to_85ns/sup_st_co2_y_85ns.csv")
sup_dfx100 =  pd.read_csv("../85_to_100ns/sup_st_co2_x_100ns.csv")
sup_dfy100 =  pd.read_csv("../85_to_100ns/sup_st_co2_y_100ns.csv")


sup_x10  = np.hstack(con_csv_bis(sup_dfx10))
sup_y10  = np.hstack(con_csv_bis(sup_dfy10))
sup_x25  = np.hstack(con_csv_bis(sup_dfx25))
sup_y25  = np.hstack(con_csv_bis(sup_dfy25))
sup_x40  = np.hstack(con_csv_bis(sup_dfx40))
sup_y40  = np.hstack(con_csv_bis(sup_dfy40))
sup_x55  = np.hstack(con_csv_bis(sup_dfx55))
sup_y55  = np.hstack(con_csv_bis(sup_dfy55))
sup_x70  = np.hstack(con_csv_bis(sup_dfx70))
sup_y70  = np.hstack(con_csv_bis(sup_dfy70))
sup_x85  = np.hstack(con_csv_bis(sup_dfx85))
sup_y85  = np.hstack(con_csv_bis(sup_dfy85))
sup_x100 = np.hstack(con_csv_bis(sup_dfx100))
sup_y100 = np.hstack(con_csv_bis(sup_dfy100))

sup_x100ns = np.concatenate((sup_x10,sup_x25,sup_x40,sup_x55,sup_x70,sup_x85,sup_x100), axis = 0)
sup_y100ns = np.concatenate((sup_y10,sup_y25,sup_y40,sup_y55,sup_y70,sup_y85,sup_y100), axis = 0)

sup_x100ns = np.hstack(sup_x100ns)
sup_y100ns = np.hstack(sup_y100ns)

nx_sh , ny_sh = 500 , 500
nx_sh1 , ny_sh1 = 501 , 501

minx = np.min(sup_x100ns) 
miny = np.min(sup_y100ns) 
maxx = np.max(sup_x100ns) 
maxy = np.max(sup_y100ns) 

xbins = np.linspace(minx,maxx,nx_sh1)
ybins = np.linspace(miny,maxy,ny_sh1)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
ret = stats.binned_statistic_2d(sup_x100ns, sup_y100ns, None, 'count', bins=[xbins, ybins],expand_binnumbers=True)

image   = np.float64(ret[0])

# Equalization

img_equ = exposure.adjust_log(image, 1)

img_eq   = denoise_tv_chambolle(img_equ, weight=0.8)  # 0.7  0.5 0.4


np.set_printoptions(threshold=sys.maxsize)


sigma_min = 1 #1 #  1
sigma_max = 8 # 4 #  16 #  16

features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max) 

features = features_func(img_eq)
 

importancess = []
rank_feat_imps = []
entrops_em = []
entrops_ne = []
entrops_hi = []
entrop_import = []

tr_entrops_em = []
tr_entrops_ne = []
tr_entrops_hi = []
tr_entrops = []

entrops = []
entrops_pred = []
wasser_d = []
nc = 16
rel_entros = []
for n,k in enumerate(comb_array_sig[:,:]):
    zeros  = np.zeros(features[:,:,0].shape)
    zeros[(features[:,:,k[1]] >  -0.005)  & (features[:,:,k[0]] > -0.005)] = 1 
    zeros[(features[:,:,k[1]] <= -0.005)  & (features[:,:,k[0]] > -0.005)] = 2 
    zeros[(features[:,:,k[1]] >= -0.005)  & (features[:,:,k[0]] < -0.005)] = 2 
    zeros[(features[:,:,k[1]] <= -0.005)  & (features[:,:,k[0]] < -0.005)] = 3 


    annotation = zeros.astype(int)

    sele1 = features[:,:,[k[0],k[1]]][(zeros == 1),:]
    sele_ann1 = annotation[np.where(zeros == 1)]
    
    sele2 = features[:,:,[k[0],k[1]]][(zeros == 2),:]
    sele_ann2 = annotation[np.where(zeros == 2)]
    
    sele3 = features[:,:,[k[0],k[1]]][(zeros == 3),:]
    sele_ann3 = annotation[np.where(zeros == 3)]
    
    s = [len(sele1),len(sele2),len(sele3)]
    s1 = len(sele1)
    s2 = len(sele2)
    s3 = len(sele3)

   
    zeros = zeros.astype(int)
    fig, ax1 = plt.subplots(1,1,figsize=(10,10))
    colors = ['blue','deepskyblue','red']
    ax1.imshow(zeros, cmap = "jet")

    plt.savefig('sup_segmented_map.'+str(k[0])+'.'+str(k[1])+'.png') 

    stot = s1+s2+s3
    print(s1,s2,s3,stot)
    result = annotation - 1

    ##### Get coordinates for residence time and check for PBC


    reshaped_labels_sigma05 = result  
    reshaped_labels_sigma05_empty = np.copy(reshaped_labels_sigma05)
    reshaped_labels_sigma05_netwo = np.copy(reshaped_labels_sigma05)
    reshaped_labels_sigma05_highs = np.copy(reshaped_labels_sigma05)
    reshaped_labels_sigma05_empty[reshaped_labels_sigma05_empty != 0] = 3
    reshaped_labels_sigma05_netwo[reshaped_labels_sigma05_netwo != 1] = 3
    reshaped_labels_sigma05_highs[reshaped_labels_sigma05_highs != 2] = 3
    
    
    conn_labels_sigma05_empty, nlab_sigma05_empty = measure.label(reshaped_labels_sigma05_empty, background = 3,return_num =  True)
    conn_labels_sigma05_netwo, nlab_sigma05_netwo = measure.label(reshaped_labels_sigma05_netwo, background = 3,return_num =  True)
    conn_labels_sigma05_highs, nlab_sigma05_highs = measure.label(reshaped_labels_sigma05_highs, background = 3,return_num =  True)
    
    # Relabel network as a continous (only label 1)
    
    conn_labels_sigma05_netwo_emp  = np.copy(conn_labels_sigma05_highs)
    conn_labels_sigma05_highs_all  = np.copy(conn_labels_sigma05_highs)
    
    
    conn_labels_sigma05_highs_all[(conn_labels_sigma05_highs_all != 0)] = 1
    
    
    conn_labels_sigma05_netwo_emp[(conn_labels_sigma05_netwo_emp != 0)] = 2
    conn_labels_sigma05_netwo_emp[(conn_labels_sigma05_netwo_emp == 0)] = 1
    conn_labels_sigma05_netwo_emp[(conn_labels_sigma05_netwo_emp == 2)] = 0
    
    
    ones = np.ones(image.shape)
    ones = ones.astype(int)
    
    coordinates_highs = boundary(conn_labels_sigma05_highs,0)[0]
    num_regions_highs = boundary(conn_labels_sigma05_highs,0)[1]
    coordinates_netwo = boundary(conn_labels_sigma05_netwo,0)[5]
    num_regions_netwo = boundary(conn_labels_sigma05_netwo,0)[1]
    coordinates_empty = boundary(conn_labels_sigma05_empty,0)[5]
    num_regions_empty = boundary(conn_labels_sigma05_empty,0)[1]
    area_empty = boundary(conn_labels_sigma05_empty,0)[4]
    area_netwo = boundary(conn_labels_sigma05_netwo,0)[4]
    
    coordinates_netwo = np.vstack(coordinates_netwo)
    coordinates_empty = np.vstack(coordinates_empty)
    
    print(coordinates_netwo.shape)
    print(coordinates_empty.shape)
    
    
    np.savetxt('sup_area_empty'+str(k[0])+'.'+str(k[1])+'.txt' ,area_empty , fmt = "%s"  )
    np.savetxt('sup_area_netwo'+str(k[0])+'.'+str(k[1])+'.txt' ,area_netwo , fmt = "%s"  )
    
    coordinates_netwo_emp = boundary(conn_labels_sigma05_netwo_emp,0)[0]
    num_regions_netwo_emp = boundary(conn_labels_sigma05_netwo_emp,0)[1]
    
    
    coordinates_highs_all = boundary(conn_labels_sigma05_highs_all,0)[0]
    num_regions_highs_all = boundary(conn_labels_sigma05_highs_all,0)[1]
    
    
    labels_all = np.copy(conn_labels_sigma05_highs)
    labels_all[(labels_all == 0)] = 1
    labels_all[(labels_all != 0)] = 1
    
    coordinates_all = boundary(labels_all,0)[0]
    
    cor_xinf  = edges_pbc(coordinates_highs,0)[0]
    cor_xsup  = edges_pbc(coordinates_highs,0)[1]
    num_xinf  = edges_pbc(coordinates_highs,0)[2]
    num_xsup  = edges_pbc(coordinates_highs,0)[3]
    
    cor_yinf  = edges_pbc(coordinates_highs,1)[0]
    cor_ysup  = edges_pbc(coordinates_highs,1)[1]
    num_yinf  = edges_pbc(coordinates_highs,1)[2]
    num_ysup  = edges_pbc(coordinates_highs,1)[3]
    
    
    pb_x = conn_pbc(cor_xinf,cor_xsup,num_xinf,num_xsup,1) # 1
    pb_y = conn_pbc(cor_yinf,cor_ysup,num_yinf,num_ysup,0) # 0
    pb = np.concatenate((pb_x,pb_y), axis = 0)
    
    keep = overlap(pb_x,pb_y)
    
    merge, merged = merge_reg_pbc(keep,pb)
    highs_mer = np.copy(conn_labels_sigma05_highs)
    
    highs_merged = relabel(merge,merged)
    
    
    num_regions_highs_merged     = boundary(highs_merged,0)[1]  
    areas_highs_merged           = boundary(highs_merged,0)[4]  
    coordinates_highs_merged     = boundary(highs_merged,0)[5]
    
    
    conn_labels_sigma05_netwo, nlab_sigma05_netwo = measure.label(conn_labels_sigma05_netwo, background = 0,return_num =  True)
    
    np.savetxt('sup_area_regions_highs_merged'+str(k[0])+'.'+str(k[1])+'.txt' ,np.vstack(areas_highs_merged)     , fmt = "%s"  )
    
    
    len_net     = len(coordinates_netwo_emp)*2
    len_net_emp = len(coordinates_netwo)*2
    
    st_coordinates_netwo_emp = np.hstack(coordinates_netwo_emp)
    st_coordinates_netwo     = np.hstack(coordinates_netwo)
    st_coordinates_empty     = np.hstack(coordinates_empty)
    
    
    len_net_emp     = st_coordinates_netwo_emp.shape[0]*2
    len_net         = st_coordinates_netwo.shape[0] # *2
    len_emp         = st_coordinates_empty.shape[0] # *2
    
    st_coordinates_netwo_emp = st_coordinates_netwo_emp.reshape(len_net_emp)
    st_coordinates_netwo     =     st_coordinates_netwo.reshape(len_net)    
    st_coordinates_empty     =     st_coordinates_empty.reshape(len_emp)    
    
    
    np.savetxt('sup_coordinates_netwo_emp'+str(k[0])+'.'+str(k[1])+'.txt' ,st_coordinates_netwo_emp  , fmt = "%s"  )
    
    tag = np.array([k[0],k[1]])


    conn_labels_sigma05_netwo_emp = pd.DataFrame(conn_labels_sigma05_netwo_emp)
    
    
    highs = pd.DataFrame(coordinates_highs_merged)
    highs.to_csv('sup_coor_high_{}.csv'.format(tag))  
    
    highs = np.zeros(image.shape)
    
    highs_conn, nlab_highs_conn = measure.label(highs, background = 0,return_num =  True)
    np.savetxt('sup_conn_highs'+str(k[0])+'.'+str(k[1])+'.txt' ,highs_conn , fmt = "%s"  )


    ##### End get coordinates for RT

