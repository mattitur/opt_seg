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
import cv2

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
import xgboost as xgb
from xgboost import XGBClassifier

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


import sys

from sklearn.model_selection import train_test_split

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



def box_vecs(coord_vecs):
    x_inf=[]
    x_sup=[]
    for i,lines in enumerate(coord_vecs):
        x_inf.append(lines.split(" ")[0])
        x_sup.append(lines.split(" ")[1])
        X_inf=np.array(x_inf)
        X_sup=np.array(x_sup)
        X_inf=X_inf.astype(float)
        X_sup=X_sup.astype(float)
        x  = X_sup - X_inf
    return(x,X_sup,X_inf)


def read_file(filename):
    ts=[]
    number_of_atoms=[]
    box_vec_x=[]
    box_vec_y=[]
    box_vec_z=[]
    linets=1
    linean=3
    linex=5
    liney=6
    linez=7
    with open(filename) as f:
         line=f.readlines()
         line2=line[9:]
         iline=iter(line)
         for i,lines in enumerate(iline):

             if i==linets:
                ts.append(lines.rstrip("\n"))
             if i==linean:
                an=lines.strip("\n")
                number_of_atoms.append(lines.rstrip("\n"))
             if i==linex:
                box_vec_x.append(lines.rstrip("\n"))
             if i==liney:
                box_vec_y.append(lines.rstrip("\n"))
             if i==linez:
                box_vec_z.append(lines.rstrip("\n"))
                linets += int(an)+9
                linean += int(an)+9
                linex  += int(an)+9
                liney  += int(an)+9
                linez  += int(an)+9

         return(ts,number_of_atoms,box_vec_x,box_vec_y,box_vec_z)


def read_coord(filename,number_of_atoms,number_of_frames):
    a_num=[]
    atom_type=[]
    x_coor=[]
    y_coor=[]
    z_coor=[]
    pos_coor=[]

    with open(filename) as f:
         lines_filter = chain.from_iterable(repeat([False]*9 + [True]*number_of_atoms, number_of_frames))
         lines = compress(f, lines_filter)
         values =  map(lambda line: line.strip(), lines) 
         for lines in values:
             a_num.append(lines.split(" ")[0])
             atom_type.append(lines.split(" ")[1])
             pos_coor.append(lines.strip().split(" ")[2:5])
         a_num=np.array(a_num).flatten()
         atom_type=np.array(atom_type).flatten()
         pos_coor=np.array(pos_coor)
         return(a_num, atom_type, pos_coor)




def sel_atoms(atom_numb, atom_type, coords, frames,dim):


    atom_numb=np.split(atom_numb,frames)
    atom_type=np.split(atom_type,frames)
    coords=np.vsplit(coords,frames)
    coords=np.array(coords)
    atom_numb=np.array(atom_numb)
    atom_type=np.array(atom_type)
    
    coords=coords.astype(float)
    atom_numb=atom_numb.astype(float)
    atom_type=atom_type.astype(float) 
    
    oxy=[]
    hydro=[]
    calcium=[]
    silicon=[]
    aluminium=[]
    oxy_h = []
    hydro_h = []
    oxy_b = []
    num_o=[]
    num_h=[]
    num_ca=[]
    num_si=[]
    num_al=[]
    num_oxy_h=[]
    num_h_h=[]
    num_oxy_b = []
    for k, (an,at,coord) in enumerate(zip(atom_numb,atom_type,coords)):
        
        for a_num, at, pos in zip(an,at,coord):
            if at == 5:    # oxy silica
               oxy.append(pos)
               num_o.append(a_num)
            if at == 5:    # h in oh
               hydro.append(pos)
               num_h.append(a_num)
            if at == 5:    # o in  oh
               calcium.append(pos)
               num_ca.append(a_num)
            if at == 5:    # si in silica
               silicon.append(pos)
               num_si.append(a_num)
            if at == 6:    # o in co2
               aluminium.append(pos)
               num_al.append(a_num)
            if at == 5:    # c in co2
               oxy_b.append(pos)
               num_oxy_b.append(a_num)
            if at == 5:
               oxy_h.append(pos)
               num_oxy_h.append(a_num)
            if at == 5:
               hydro_h.append(pos)
               num_h_h.append(a_num)
    
    
    oxy=np.array(oxy)
    num_o=np.array(num_o)
    hydro=np.array(hydro)
    num_h=np.array(num_h)
    calcium=np.array(calcium)
    num_ca=np.array(num_ca)
    silicon=np.array(silicon)
    num_si=np.array(num_si)
    aluminium=np.array(aluminium)
    num_al=np.array(num_al)
    oxy_b=np.array(oxy_b)
    num_oxy_b=np.array(num_oxy_b)
    oxy_h=np.array(oxy_h)
    num_oxy_h=np.array(num_oxy_h)
    hydro_h=np.array(hydro_h)
    num_h_h=np.array(num_h_h)
    
    
    
    num_o = num_o.reshape(len(num_o),1)
    oxy=oxy.reshape(len(oxy[:,0]),3)
    num_h = num_h.reshape(len(num_h),1)
    hydro=hydro.reshape(len(hydro[:,0]),3)
    num_ca = num_ca.reshape(len(num_ca),1)
    calcium=calcium.reshape(len(calcium[:,0]),3)
    num_al = num_al.reshape(len(num_al),1)
    aluminium=aluminium.reshape(len(aluminium[:,0]),3)
    num_si = num_si.reshape(len(num_si),1)
    silicon=silicon.reshape(len(silicon[:,0]),3)
    num_oxy_b = num_oxy_b.reshape(len(num_oxy_b),1)
    oxy_b=oxy_b.reshape(len(oxy_b[:,0]),3)
    num_oxy_h = num_oxy_h.reshape(len(num_oxy_h),1)
    oxy_h=oxy_h.reshape(len(oxy_h[:,0]),3)
    num_h_h = num_h_h.reshape(len(num_h_h),1)
    hydro_h=hydro_h.reshape(len(hydro_h[:,0]),3)
    
    
      
    hydro=np.vsplit(hydro,frames)       
    oxy=np.vsplit(oxy,frames)
    hydro=np.array(hydro)
    oxy=np.array(oxy)
    num_o=np.split(num_o,frames)
    num_h=np.split(num_h,frames)
    
    hydro_h=np.vsplit(hydro_h,frames)       
    oxy_h=np.vsplit(oxy_h,frames)
    hydro_h=np.array(hydro_h)
    oxy_h=np.array(oxy_h)
    num_oxy_h=np.split(num_oxy_h,frames)
    num_h_h=np.split(num_h_h,frames)
    
    
    silicon=np.vsplit(silicon,frames)       
    calcium=np.vsplit(calcium,frames)
    silicon=np.array(silicon)
    calcium=np.array(calcium)
    aluminium=np.vsplit(aluminium,frames)
    aluminium=np.array(aluminium)
    
    num_ca=np.split(num_ca,frames)
    num_si=np.split(num_si,frames)
    num_al=np.split(num_al,frames)
    
    oxy_b=np.vsplit(oxy_b,frames)
    oxy_b=np.array(oxy_b)
    num_oxy_b=np.split(num_oxy_b,frames)


    return(num_o,oxy,num_h,hydro,num_ca,calcium,num_al,aluminium,num_si,silicon,num_oxy_b,oxy_b,num_oxy_h,oxy_h,num_h_h,hydro_h)


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
    start =    0 #  10   #   10   # 10     # 10      #10     # 10   # 0
    fin   =  400 #  250  #  250   # 80     # 230     #30     # 410  # 200
    end   =  9500 #  9610 #  730   # 20     # 9690   #970    # 9530 # 300
    final =  9900 #  9850 #  970   # 90     # 9910   #990    # 9930 # 500
    step  =  400 #  240  #  240   # 2  # 5 # 220   #2      # 40   # 20
    
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

            print(mask0.size)

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

def con_csv(df):
    co2_x = []
    for row in df["0"]:
        row = row.replace('\n','')
        row = row.replace("[",'')
        row = row.replace("]",'')
        row = row.replace("   ",' ')
        row = row.replace("  ",' ')

        row = row.strip().split(' ') # , '  ')  # .split('  ') 
        arr = np.array(row)
        arr = arr.astype(float)
        co2_x.append(arr)
    return(co2_x)



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


inf_x10  = con_csv_bis(inf_dfx10)
inf_y10  = con_csv_bis(inf_dfy10)
inf_x25  = con_csv_bis(inf_dfx25)
inf_y25  = con_csv_bis(inf_dfy25)
inf_x40  = con_csv_bis(inf_dfx40)
inf_y40  = con_csv_bis(inf_dfy40)
inf_x55  = con_csv_bis(inf_dfx55)
inf_y55  = con_csv_bis(inf_dfy55)
inf_x70  = con_csv_bis(inf_dfx70)
inf_y70  = con_csv_bis(inf_dfy70)
inf_x85  = con_csv_bis(inf_dfx85)
inf_y85  = con_csv_bis(inf_dfy85)
inf_x100 = con_csv_bis(inf_dfx100)
inf_y100 = con_csv_bis(inf_dfy100)

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


sup_x10  = con_csv_bis(sup_dfx10)
sup_y10  = con_csv_bis(sup_dfy10)
sup_x25  = con_csv_bis(sup_dfx25)
sup_y25  = con_csv_bis(sup_dfy25)
sup_x40  = con_csv_bis(sup_dfx40)
sup_y40  = con_csv_bis(sup_dfy40)
sup_x55  = con_csv_bis(sup_dfx55)
sup_y55  = con_csv_bis(sup_dfy55)
sup_x70  = con_csv_bis(sup_dfx70)
sup_y70  = con_csv_bis(sup_dfy70)
sup_x85  = con_csv_bis(sup_dfx85)
sup_y85  = con_csv_bis(sup_dfy85)
sup_x100 = con_csv_bis(sup_dfx100)
sup_y100 = con_csv_bis(sup_dfy100)

nx_sh , ny_sh = 500 , 500
nx_sh1 , ny_sh1 = 501 , 501

inf_x100ns = np.concatenate((inf_x10,inf_x25,inf_x40,inf_x55,inf_x70,inf_x85,inf_x100), axis = 0)
inf_y100ns = np.concatenate((inf_y10,inf_y25,inf_y40,inf_y55,inf_y70,inf_y85,inf_y100), axis = 0)
inf_x100ns = np.hstack(inf_x100ns)
inf_y100ns = np.hstack(inf_y100ns)
inf_minx = np.min(inf_x100ns) #  + 0.5
inf_miny = np.min(inf_y100ns) #  + 0.5
inf_maxx = np.max(inf_x100ns) #  - 0.5
inf_maxy = np.max(inf_y100ns) #  - 0.5
inf_xbins = np.linspace(inf_minx,inf_maxx,nx_sh1)
inf_ybins = np.linspace(inf_miny,inf_maxy,ny_sh1)


sup_x100ns = np.concatenate((sup_x10,sup_x25,sup_x40,sup_x55,sup_x70,sup_x85,sup_x100), axis = 0)
sup_y100ns = np.concatenate((sup_y10,sup_y25,sup_y40,sup_y55,sup_y70,sup_y85,sup_y100), axis = 0)
sup_x100ns = np.hstack(sup_x100ns)
sup_y100ns = np.hstack(sup_y100ns)
sup_minx = np.min(sup_x100ns) #  + 0.5
sup_miny = np.min(sup_y100ns) #  + 0.5
sup_maxx = np.max(sup_x100ns) #  - 0.5
sup_maxy = np.max(sup_y100ns) #  - 0.5
sup_xbins = np.linspace(sup_minx,sup_maxx,nx_sh1)
sup_ybins = np.linspace(sup_miny,sup_maxy,ny_sh1)


np.set_printoptions(threshold=sys.maxsize)


#exit()
sup_dfx10  = pd.read_csv("../0_to_10ns/wind_sup_st_co2_x_10.csv")
sup_dfy10  = pd.read_csv("../0_to_10ns/wind_sup_st_co2_y_10.csv")
sup_dfx25 =  pd.read_csv("../10_to_25ns/wind_sup_st_co2_x_25.csv")
sup_dfy25 =  pd.read_csv("../10_to_25ns/wind_sup_st_co2_y_25.csv")
sup_dfx40 =  pd.read_csv("../25_to_40ns/wind_sup_st_co2_x_40.csv")
sup_dfy40 =  pd.read_csv("../25_to_40ns/wind_sup_st_co2_y_40.csv")
sup_dfx55 =  pd.read_csv("../40_to_55ns/wind_sup_st_co2_x_55.csv")
sup_dfy55 =  pd.read_csv("../40_to_55ns/wind_sup_st_co2_y_55.csv")
sup_dfx70 =  pd.read_csv("../55_to_70ns/wind_sup_st_co2_x_70.csv")
sup_dfy70 =  pd.read_csv("../55_to_70ns/wind_sup_st_co2_y_70.csv")
sup_dfx85 =  pd.read_csv("../70_to_85ns/wind_sup_st_co2_x_85.csv")
sup_dfy85 =  pd.read_csv("../70_to_85ns/wind_sup_st_co2_y_85.csv")
sup_dfx100 = pd.read_csv("../85_to_100ns/wind_sup_st_co2_x_100.csv")
sup_dfy100 = pd.read_csv("../85_to_100ns/wind_sup_st_co2_y_100.csv")



sup_dfx10 = con_csv_bis(sup_dfx10)
sup_dfy10 = con_csv_bis(sup_dfy10)
sup_dfx25 = con_csv_bis(sup_dfx25)
sup_dfy25 = con_csv_bis(sup_dfy25)
sup_dfx40 = con_csv_bis(sup_dfx40)
sup_dfy40 = con_csv_bis(sup_dfy40)
sup_dfx55 = con_csv_bis(sup_dfx55)
sup_dfy55 = con_csv_bis(sup_dfy55)
sup_dfx70 = con_csv_bis(sup_dfx70)
sup_dfy70 = con_csv_bis(sup_dfy70)
sup_dfx85 = con_csv_bis(sup_dfx85)
sup_dfy85 = con_csv_bis(sup_dfy85)
sup_dfx100= con_csv_bis(sup_dfx100)
sup_dfy100= con_csv_bis(sup_dfy100)



co2_x100_sup = np.concatenate((sup_dfx10,sup_dfx25,sup_dfx40,sup_dfx55,sup_dfx70,sup_dfx85,sup_dfx100), axis = 0)
co2_y100_sup = np.concatenate((sup_dfy10,sup_dfy25,sup_dfy40,sup_dfy55,sup_dfy70,sup_dfy85,sup_dfy100), axis = 0)



inf_dfx10  = pd.read_csv("../0_to_10ns/wind_inf_st_co2_x_10.csv")
inf_dfy10  = pd.read_csv("../0_to_10ns/wind_inf_st_co2_y_10.csv")
inf_dfx25 =  pd.read_csv("../10_to_25ns/wind_inf_st_co2_x_25.csv")
inf_dfy25 =  pd.read_csv("../10_to_25ns/wind_inf_st_co2_y_25.csv")
inf_dfx40 =  pd.read_csv("../25_to_40ns/wind_inf_st_co2_x_40.csv")
inf_dfy40 =  pd.read_csv("../25_to_40ns/wind_inf_st_co2_y_40.csv")
inf_dfx55 =  pd.read_csv("../40_to_55ns/wind_inf_st_co2_x_55.csv")
inf_dfy55 =  pd.read_csv("../40_to_55ns/wind_inf_st_co2_y_55.csv")
inf_dfx70 =  pd.read_csv("../55_to_70ns/wind_inf_st_co2_x_70.csv")
inf_dfy70 =  pd.read_csv("../55_to_70ns/wind_inf_st_co2_y_70.csv")
inf_dfx85 =  pd.read_csv("../70_to_85ns/wind_inf_st_co2_x_85.csv")
inf_dfy85 =  pd.read_csv("../70_to_85ns/wind_inf_st_co2_y_85.csv")
inf_dfx100 = pd.read_csv("../85_to_100ns/wind_inf_st_co2_x_100.csv")
inf_dfy100 = pd.read_csv("../85_to_100ns/wind_inf_st_co2_y_100.csv")

inf_dfx10 = con_csv_bis(inf_dfx10)
inf_dfy10 = con_csv_bis(inf_dfy10)
inf_dfx25 = con_csv_bis(inf_dfx25)
inf_dfy25 = con_csv_bis(inf_dfy25)
inf_dfx40 = con_csv_bis(inf_dfx40)
inf_dfy40 = con_csv_bis(inf_dfy40)
inf_dfx55 = con_csv_bis(inf_dfx55)
inf_dfy55 = con_csv_bis(inf_dfy55)
inf_dfx70 = con_csv_bis(inf_dfx70)
inf_dfy70 = con_csv_bis(inf_dfy70)
inf_dfx85 = con_csv_bis(inf_dfx85)
inf_dfy85 = con_csv_bis(inf_dfy85)
inf_dfx100= con_csv_bis(inf_dfx100)
inf_dfy100= con_csv_bis(inf_dfy100)



co2_x100_inf = np.concatenate((inf_dfx10,inf_dfx25,inf_dfx40,inf_dfx55,inf_dfx70,inf_dfx85,inf_dfx100), axis = 0)
co2_y100_inf = np.concatenate((inf_dfy10,inf_dfy25,inf_dfy40,inf_dfy55,inf_dfy70,inf_dfy85,inf_dfy100), axis = 0)



def bin_single(co2_x,co2_y,xbins,ybins):

    retini_xy = []
    retini_x = []
    retini_y = []
    retini_len = []
    
    for n, (xi,yi) in enumerate(zip(co2_x,co2_y)):
        #print("x", xi)
        if len(xi) > 1:
           ret_in = stats.binned_statistic_2d(xi[::1], yi[::1], None, 'count', bins=[xbins, ybins],expand_binnumbers=True)

        else:
           ret_in = stats.binned_statistic_2d(xi, yi, None, 'count', bins=[xbins, ybins],expand_binnumbers=True)

        x_bins = ret_in[3][0]
        y_bins = ret_in[3][1]
    
        x_bins = x_bins.reshape(len(x_bins),1)
        y_bins = y_bins.reshape(len(y_bins),1)
        xy_bins = np.concatenate((x_bins,y_bins), axis = 1)
    
        retini_x.append(x_bins)
        retini_y.append(y_bins)
        retini_xy.append(xy_bins)
        retini_len.append(len(x_bins))
    
    retini_x_arr = np.array(retini_x)
    retini_y_arr = np.array(retini_y)
    retini_xy_arr = np.array(retini_xy)

    return(retini_x_arr,retini_y_arr,retini_xy_arr,retini_len,ret_in)

sup_retini_x_arr10    =  bin_single(co2_x100_sup,co2_y100_sup, sup_xbins,sup_ybins)[0]
sup_retini_y_arr10    =  bin_single(co2_x100_sup,co2_y100_sup, sup_xbins,sup_ybins)[1]
sup_retini_xy_arr10   =  bin_single(co2_x100_sup,co2_y100_sup, sup_xbins,sup_ybins)[2]
sup_retini_len10      =  bin_single(co2_x100_sup,co2_y100_sup, sup_xbins,sup_ybins)[3]

sup_highs = pd.DataFrame(sup_retini_xy_arr10)
sup_highs.to_csv('sup_retini_xy_arr100_winds.csv')  


inf_retini_x_arr10    =  bin_single(co2_x100_inf,co2_y100_inf, inf_xbins,inf_ybins)[0]
inf_retini_y_arr10    =  bin_single(co2_x100_inf,co2_y100_inf, inf_xbins,inf_ybins)[1]
inf_retini_xy_arr10   =  bin_single(co2_x100_inf,co2_y100_inf, inf_xbins,inf_ybins)[2]
inf_retini_len10      =  bin_single(co2_x100_inf,co2_y100_inf, inf_xbins,inf_ybins)[3]

inf_highs = pd.DataFrame(inf_retini_xy_arr10)
inf_highs.to_csv('inf_retini_xy_arr100_winds.csv')  


#print(np.min(np.hstack(retini_x_arr10)),np.max(np.hstack(retini_y_arr10)))
