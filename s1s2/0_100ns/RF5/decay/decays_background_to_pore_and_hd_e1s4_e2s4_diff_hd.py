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
from sklearn.metrics import r2_score

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


import sys

from sklearn.model_selection import train_test_split

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter

import matplotlib.ticker as mtick
from math import e


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

def read_rdf_bis(filename):
    rdf=[]
    with open(filename) as f:
         lines=f.readlines()
         for line in lines:
             l = line.strip().split("\n")
             rdf.append(l)
         le = int(len(rdf)/2)
         rdf=np.array(rdf).reshape(1,le,2)    
         rdf = rdf.astype(float)	 
         #print(rdf[0][0])
         #rdf = list(rdf)
         return(rdf)    



################## CV2 clustering
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

def con_csv_highs(df):
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
        #print(arr)
        l2 = int(len(arr)/2)
        arr = arr.reshape(l2,2)
        arr = arr.astype(float)
        co2_x.append(arr)
    co2_x = np.array(co2_x, dtype=object)

        #print(arr.shape)
    return(co2_x)

def ads_in_segs(coors,retini_xy_arr,treshold_noise):
    en_ind_w = []
    ind_w = []
    ind_w_no_last = []
    ind_w_no_last_sum = []

    n_of_gas = []
    ngas_no_des_st = []
    ind_true = []
    ind_w_no_ones = []
    len_des_st = []
    for n,rt in enumerate(coors): 
    #for n,rt in enumerate(coors[0:1:1]): 

        en_rt_trc = []
        rt_trc = []
        rt_trc_no_last = []
        rt_trc_no_last_sum = []
        rt_trc_no_ones = []
        ngas_no_des = []
        len_des = []
        for en,rti in enumerate(retini_xy_arr[:]):
            condition = (rti==rt[:,None]).all(2).any(0)
            condition2 = condition  # [0:-1:2]
            condition2 = np.multiply(condition2,1)
 
            ind_con = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2]
            #print(condition2,ind_con,np.sum(ind_con))
            rt_trc.append(ind_con)
            if (ind_con[:] > treshold_noise).any():
               rt_trc_no_last.append(ind_con[:][ind_con[:]>treshold_noise])
               rt_trc_no_last_sum.append(np.sum(ind_con[:][ind_con[:]>treshold_noise]))
               #print(ind_con[:][ind_con[:]>treshold_noise])
               #print(np.sum(ind_con[:][ind_con[:]>treshold_noise]))
     
            else:
               rt_trc_no_last.append(np.array([0]))
               rt_trc_no_last_sum.append(np.array([0]))
 
            en_rt_trc.append(np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[1:-1:2])
            ngas_no_des.append(condition2)
            len_des.append(len(condition2))

            if len(condition) > 0:

               ind_no_ones = np.where(ind_con > 0)
               ind_con_no_ones = ind_con[ind_no_ones]
               rt_trc_no_ones.append(ind_con_no_ones)

            else:

               ind_con_no_ones = ind_con
               rt_trc_no_ones.append(ind_con_no_ones)
        print(n,len(rt_trc))
        ind_w.append(rt_trc)
        ind_w_no_last.append(np.hstack(rt_trc_no_last))
        ind_w_no_last_sum.append(np.hstack(rt_trc_no_last_sum))


        en_ind_w.append(en_rt_trc)
        ngas_no_des_st.append(ngas_no_des)

    ind_w_no_last = np.array(ind_w_no_last)
    ind_w_no_last_sum = np.array(ind_w_no_last_sum)
 
    return(ind_w,ngas_no_des_st,en_ind_w,ind_w_no_last,ind_w_no_last_sum)           

def ads_in_segs_net(coors,retini_xy_arr,treshold_noise,gap):
    en_ind_w = []
    ind_w = []
    ind_w_no_last = []
    ind_w_last = []


    n_of_gas = []
    ngas_no_des_st = []
    ind_true = []
    ind_w_no_ones = []
    len_des_st = []
    for n,rt in enumerate(coors): 
        en_rt_trc = []
        rt_trc = []
        rt_trc_no_last = []
        rt_trc_last = []

        rt_trc_no_ones = []
        ngas_no_des = []
        len_des = []
        for en,rti in enumerate(retini_xy_arr):
            condition = (rti[::gap]==rt[:,None]).all(2).any(0)
            condition2 = condition  # [0:-1:2]
            condition2 = np.multiply(condition2,1)
            #print("rti",en,rti)
            #print("rt",n,rt)
            #print("con",en,condition2)

            ind_con = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2]
            if len(ind_con) > 0:
               rt_trc.append(ind_con)
            if len(ind_con) == 0:
               rt_trc.append(np.array([0]))
               #print(rt_trc)
            #print("ind_con",en,ind_con)
            if (ind_con[:] > treshold_noise).any():
               rt_trc_no_last.append(ind_con[:-1][ind_con[:-1]>treshold_noise])
               if (condition2[-1] == 1) and (ind_con[-1] > treshold_noise):
                  rt_trc_last.append(ind_con[-1])
               else:
                  rt_trc_last.append(0)
                  #print("ind_con",en,ind_con,ind_con[-1])
            else:
               rt_trc_no_last.append(np.array([0]))
               rt_trc_last.append(0)
 
            ngas_no_des.append(condition2)
            len_des.append(len(condition2))
            #print("coord,mol", n, en,datetime.now() )
        #print(n,len(rt_trc))
        ind_w.append(rt_trc)
        #print(ind_w)
        ind_w_no_last.append(np.hstack(rt_trc_no_last))
        ind_w_last.append(rt_trc_last)
        ngas_no_des_st.append(ngas_no_des)
    return(ind_w,ngas_no_des_st,ind_w_no_last,ind_w_last)           

def fetch(arr):
    arr1 = np.sum(arr[1:])
    return(arr1)


def noise_off(rt_trc_highs_split,en_rt_trc_highs_split,treshold_noise ):
    fil = []
    consec = []
    consec_h = []
    consec_sum = []
    consec_h_sum = []
    consec_netwo = []
    for isl,hol in zip(rt_trc_highs_split,en_rt_trc_highs_split):
        fil_mol = []
        cons = []
        cons_h = []
        cons_sum = []
        cons_h_sum = []
        cons_netwo = []
        if len(isl) > 0:
           for mol,hl in zip(isl,hol):
               ind = np.where(hl > treshold_noise)
               ind_last = ind[0] #  np.append(ind[0],-1)
               con   = np.split(hl,ind_last)
               con_h = np.split(mol,ind_last)
    
               con_sum   = np.hstack(map(np.sum,con  ))
               con_h_sum = np.hstack(map(np.sum,con_h))
    
               con_sum_m   = np.hstack(map(fetch,con  ))
               con_sum_m[0]   = con_sum[0]  
    
               net_emp = hl[ind[0]]
               fil_mol.append(ind[0])
    
               cons.append(con) 
               cons_h.append(con_h) 
    
               cons_sum.append(con_sum_m) 
               cons_h_sum.append(con_h_sum) 
               cons_netwo.append(net_emp)
    
        fil.append(fil_mol)
        consec_sum.append(cons_sum)
        consec_h_sum.append(cons_h_sum)
        consec.append(cons)
        consec_h.append(cons_h)
        consec_netwo.append(cons_netwo)
    return(consec_h_sum,consec_sum)

def sum_gaps(consec_h_sum,consec_sum):
    highs_sum_all = []
    highs_sum_all_st = []
    for con_h,con in zip(consec_h_sum,consec_sum):
        highs_sum = []
        h_jump = []
        for ch,c in zip(con_h,con):
            #print("ch_c_sizes",ch.shape,c.shape)
            highs_whole = ch + c 
            if ((highs_whole[0] == 0) and (highs_whole.shape[0] == 1)):
               h_jump.append(highs_whole)
            else:
               highs_sum.append(highs_whole)
        highs_sum_all.append(highs_sum)
        highs_sum_all_st.append(np.hstack(highs_sum))
    highs_sum_all_st = np.array(highs_sum_all_st)
    return(highs_sum_all_st)


np.set_printoptions(threshold=sys.maxsize)

c  = "black"
c1 = "red"
c2 = "white"



def f2(x, a, b, c):
    return c*(np.exp(-x/a)) + (1-c)*np.exp(-x/b)


def f(x, b):
    return  np.exp(-x/b)    

t = np.arange(0,400,1)

def decay_single_new(gas_surf, start, fin, start_fit,siz):
    gas_surf_sliced = []
    gas_surf_sliced_fit = []
    for en,left_surf in enumerate(gas_surf):
        print(en)
        gas_surf_sliced.append(left_surf)
        gas_surf_sliced_fit.append(left_surf[start_fit:])   
    gas_surf_sliced_arr = np.hstack(gas_surf_sliced)
    gas_surf_sliced_arr_fit = np.hstack(gas_surf_sliced_fit)


    bin_centers_rep = np.tile(bin_centers[:-1][:],siz)
    bin_centers_rep_fit = np.tile(bin_centers[start_fit:-1][:],siz)
   
     
    x_sort     = np.sort(bin_centers_rep)
    ind_sort   = np.argsort(bin_centers_rep)
    y_sort     = gas_surf_sliced_arr[ind_sort]

    x_sort_fit     = np.sort(bin_centers_rep_fit)
    ind_sort_fit   = np.argsort(bin_centers_rep_fit)
    y_sort_fit     = gas_surf_sliced_arr_fit[ind_sort_fit]
    y_sort_fit     = np.nan_to_num(y_sort_fit)

    #popt, pcov = curve_fit(f, x_sort, y_sort,absolute_sigma=True)
    popt, pcov = curve_fit(f, x_sort_fit, y_sort_fit,absolute_sigma=True)
    #yfit = f(x_sort, *popt)
    pcov = np.sqrt(np.diag(pcov))
    yfit = f(x_sort_fit, *popt)
    print("popt:",popt[0])
    print("pcov:",pcov[0])

    r2 = r2_score(y_sort_fit,yfit)
    print("r2",r2)



    fig, ax1 = plt.subplots(1,1,figsize=(10,10), sharey=True)
    print(len(y_sort))
    print(y_sort.shape) 
   
    y_sort_tr = np.transpose(y_sort.reshape(400,siz))
    ax1.boxplot(list(np.transpose(y_sort_tr[:,:])), notch=True, patch_artist=True,
            boxprops=dict(color=c,facecolor = c2, linewidth = 0.5),
            capprops=dict(color=c,linewidth = 0.5),
            whiskerprops=dict(color=c,linewidth = 0.1),
            flierprops=dict(markeredgecolor=c1,markerfacecolor = c1, marker = "x", markersize=1),
            medianprops=dict(linestyle = "dashed"),zorder = 2, positions=np.arange(0,400))
    #ax1.plot(x_sort, yfit, linewidth = 0.75, label = "fit eq. 6", zorder  = 15)
    ax1.plot(x_sort_fit, yfit, linewidth = 0.75, label = "fit eq. 6", zorder  = 15)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylim(bottom=0.01)  
    ax1.set_xlim([0.0, 201])  
    plt.yscale("log", base = e) #, **kwargs)
    ax1.set_title('Semilog plot')
    ax1.set_yscale('log')
    ax1.set_yticks([0.01, 0.1, 1])
    ax1.get_yaxis().set_major_formatter(mtick.ScalarFormatter())
    ax1.set_ylim(bottom=0.01)    
    ax1.set_ylim(top=1.01)    
    ax1.set_xticks(ax1.get_xticks()[0:201:10])
    labels  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    ax1.set_xticklabels(labels , fontsize = 8)
    ax1.set_xlabel("time (ps)")
    ax1.set_ylabel("N/N$_0$"  )
    #popt = pd.DataFrame(popt)
    #pcon = pd.DataFrame(pcon)
    #r2   = pd.DataFrame(r2)
    
    #return(ax1,popt,pcon,r2)
    param = np.array([popt[0],pcov[0],r2])
    #df_param = pd.DataFrame(param)
    return(ax1,param)
 
 

def decay_double_new(gas_surf, start, fin, start_fit,siz):
    gas_surf_sliced = []
    gas_surf_sliced_fit = []
    for en,left_surf in enumerate(gas_surf):
        print(en)
        gas_surf_sliced.append(left_surf)
        gas_surf_sliced_fit.append(left_surf[start_fit:])   
    gas_surf_sliced_arr = np.hstack(gas_surf_sliced)
    gas_surf_sliced_arr_fit = np.hstack(gas_surf_sliced_fit)


    bin_centers_rep = np.tile(bin_centers[:-1][:],siz)
    bin_centers_rep_fit = np.tile(bin_centers[start_fit:-1][:],siz)
   
     
    x_sort     = np.sort(bin_centers_rep)
    ind_sort   = np.argsort(bin_centers_rep)
    y_sort     = gas_surf_sliced_arr[ind_sort]

    x_sort_fit     = np.sort(bin_centers_rep_fit)
    ind_sort_fit   = np.argsort(bin_centers_rep_fit)
    y_sort_fit     = gas_surf_sliced_arr_fit[ind_sort_fit]
    y_sort_fit     = np.nan_to_num(y_sort_fit)

    #popt, pcov = curve_fit(f, x_sort, y_sort,absolute_sigma=True)
    popt, pcov = curve_fit(f2, x_sort_fit, y_sort_fit,bounds=(0,[float("inf"),float("inf"),1]),absolute_sigma=True)
    pcov = np.sqrt(np.diag(pcov))
    #yfit = f2(x_sort, *popt)
    yfit = f2(x_sort_fit, *popt)
    print("popt:",popt)
    print("pcov:",np.sqrt(np.diag(pcov)))

    r2 = r2_score(y_sort_fit,yfit)
    print("r2",r2)



    fig, ax1 = plt.subplots(1,1,figsize=(10,10), sharey=True)
    print(len(y_sort))
    print(y_sort.shape) 
   
    y_sort_tr = np.transpose(y_sort.reshape(400,siz))
    ax1.boxplot(list(np.transpose(y_sort_tr[:,:])), notch=True, patch_artist=True,
            boxprops=dict(color=c,facecolor = c2, linewidth = 0.5),
            capprops=dict(color=c,linewidth = 0.5),
            whiskerprops=dict(color=c,linewidth = 0.1),
            flierprops=dict(markeredgecolor=c1,markerfacecolor = c1, marker = "x", markersize=1),
            medianprops=dict(linestyle = "dashed"),zorder = 2, positions=np.arange(0,400))
    #ax1.plot(x_sort, yfit, linewidth = 0.75, label = "fit eq. 6", zorder  = 15)
    ax1.plot(x_sort_fit, yfit, linewidth = 0.75, label = "fit eq. 6", zorder  = 15)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylim(bottom=0.01)  
    ax1.set_xlim([0.0, 201])  
    plt.yscale("log", base = e) #, **kwargs)
    ax1.set_title('Semilog plot')
    ax1.set_yscale('log')
    ax1.set_yticks([0.01, 0.1, 1])
    ax1.get_yaxis().set_major_formatter(mtick.ScalarFormatter())
    ax1.set_ylim(bottom=0.01)    
    ax1.set_ylim(top=1.01)    
    ax1.set_xticks(ax1.get_xticks()[0:201:10])
    labels  = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    ax1.set_xticklabels(labels , fontsize = 8)
    ax1.set_xlabel("time (ps)")
    ax1.set_ylabel("N/N$_0$"  )
    #popt = pd.DataFrame(popt)
    #pcon = pd.DataFrame(pcon)
    #r2   = pd.DataFrame(r2)
    param = np.array([popt[0],popt[1],popt[2],pcov[0],pcov[1],pcov[2],r2])
    #df_param = pd.DataFrame(param)
    return(ax1,param)
 
 


def decay_single_all_smooth(gas_surfaces, start, fin, start_fit,siz):

    i,j=0,0
    PLOTS_PER_ROW = int(len(inf_area_high)/4)
    fig, axs = plt.subplots(PLOTS_PER_ROW,PLOTS_PER_ROW, figsize=(20, 20))
    popts = []
    pcons = []
    r2s = []

    for gas_surf in gas_surfaces:
        gas_surf_sliced = []
        gas_surf_sliced_fit = []
        for st,fi in zip(start,fin):
            ix = gas_surf[st:fi]        
            ix = np.hstack(ix)
            hist_surf = np.histogram(ix[ix > 0], bins=bin_centers)
            bin_surf = np.round(hist_surf[1])
            p_surf = hist_surf[0]
            comul_surf = np.sum(p_surf)
            cum_surf = np.cumsum(p_surf)
            left_surf = (comul_surf - cum_surf)/comul_surf
            gas_surf_sliced.append(left_surf)
            gas_surf_sliced_fit.append(left_surf[start_fit:])   
        
        gas_surf_sliced_arr = np.hstack(gas_surf_sliced)
        gas_surf_sliced_arr_fit = np.hstack(gas_surf_sliced_fit)


        bin_centers_rep = np.tile(bin_centers[:-1][:],siz)
        bin_centers_rep_fit = np.tile(bin_centers[start_fit:-1][:],siz)
   
         
        x_sort     = np.sort(bin_centers_rep)
        ind_sort   = np.argsort(bin_centers_rep)
        y_sort     = gas_surf_sliced_arr[ind_sort]

        x_sort_fit     = np.sort(bin_centers_rep_fit)
        ind_sort_fit   = np.argsort(bin_centers_rep_fit)
        y_sort_fit     = gas_surf_sliced_arr_fit[ind_sort_fit]

        #popt, pcov = curve_fit(f, x_sort, y_sort,absolute_sigma=True)
        popt, pcov = curve_fit(f, x_sort_fit, y_sort_fit,absolute_sigma=True)
        p = np.round_(np.array([popt,np.sqrt(np.diag(pcov))]), decimals = 2)
        #yfit = f(x_sort, *popt)
        yfit = f(x_sort_fit, *popt)
        pcov = np.sqrt(np.diag(pcov))
        print("popt:",popt)
        print("pcov:",pcov)
        r2 = r2_score(y_sort_fit,yfit)
        print("r2",r2)

        
        y_sort_tr = np.transpose(y_sort.reshape(400,siz))
        axs[i][j].boxplot(list(np.transpose(y_sort_tr[:,:])), notch=True, patch_artist=True,
                boxprops=dict(color=c,facecolor = c2, linewidth = 0.5),
                capprops=dict(color=c,linewidth = 0.5),
                whiskerprops=dict(color=c,linewidth = 0.1),
                flierprops=dict(markeredgecolor=c1,markerfacecolor = c1, marker = "x", markersize=1),
                medianprops=dict(linestyle = "dashed"),zorder = 2, positions=np.arange(0,400))
        axs[i][j].plot(x_sort_fit, yfit, linewidth = 0.75, label = "fit eq. 6", zorder  = 15)
        axs[i][j].xaxis.set_major_formatter(ScalarFormatter())
        axs[i][j].set_xlim([0.0, 121])  
        plt.yscale("log", base = e) #, **kwargs)
        axs[i][j].set_yscale('log')
        axs[i][j].set_yticks([0.01, 0.1, 1])
        axs[i][j].get_yaxis().set_major_formatter(mtick.ScalarFormatter())
        axs[i][j].set_ylim(bottom=0.01)    
        axs[i][j].set_ylim(top=1.01)    
        axs[i][j].set_xticks(axs[i][j].get_xticks()[0:121:10])
        labels  = [0,10,20,30,40,50,60,70,80,90,100,110,120]
        axs[i][j].set_xticklabels(labels , fontsize = 8)
        axs[i][j].set_xlabel("time (ps)")
        axs[i][j].set_ylabel("N/N$_0$"  )
        print(p.shape)
        #axs[i][j].annotate(str(math.tau)+str("=")+p[0]+str("(")+p[1]+str(")"),(60,0.6))
        axs[i][j].annotate(('\u03C4  = {}'.format(p[0][0]),"SD = {}".format(p[1][0])), xy=(0.4, 0.92), xycoords='axes fraction')
        popts.append(p[0][0])
        pcons.append(p[1][0])
        r2s.append(r2)

        # Update index images
        j+=1
        if j%PLOTS_PER_ROW==0:
            i+=1
            j=0

    popts = pd.DataFrame(popts)
    pcons = pd.DataFrame(pcons)
    r2s   = pd.DataFrame(r2s)



    return(plt,popts,pcons,r2s)





print(startTime)

########## Upper surface
########

inf_mol_x_wind = pd.read_csv("inf_mol_x_wind.csv").to_numpy()[:,1]
inf_fin = np.cumsum(inf_mol_x_wind)
inf_start = np.append(0,inf_fin[:-1])

sup_mol_x_wind = pd.read_csv("sup_mol_x_wind.csv").to_numpy()[:,1]
sup_fin = np.cumsum(sup_mol_x_wind)
sup_start = np.append(0,sup_fin[:-1])



def desorb(gas_ads,net_coor,start,fin):

    al = [[str(ele) for ele in net_coor[0]]][-1] 
    res2 = [[str(ele) for ele in gas_surf] for gas_surf in gas_surfaces]

    allll_to_lay = []
    timl_to_lay = []
    allll_to_hig = []
    timl_to_hig = []

    for en,(st,fi) in enumerate(zip(start,fin)):
        alll_to_lay = []
        tim_to_lay = []
        alll_to_hig = []
        tim_to_hig = []

        alll = []
        tim = []
        print(en)
        for r in res2[st:fi]:
            ads = np.isin(r,al)
            add = 400 - len(ads)
            add = np.full((add,),"False")
            ads_add = np.concatenate((ads, add))
        
            if ads[0] == True: 
               condition = ads
               ind_con = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2]
               print(ind_con)
               print(ind_con.shape[0])
               if ind_con.shape[0] == 1:
                  alll_to_lay.append(ads_add)
                  tim_to_lay.append(ind_con[0])
               if ind_con.shape[0] > 1:
                  alll_to_hig.append(ads_add)
                  tim_to_hig.append(ind_con[0])


        alll_ar_to_lay = np.array(alll_to_lay)
        allll_to_lay.append(alll_to_lay)
        timl_to_lay.append(tim_to_lay)

        alll_ar_to_hig = np.array(alll_to_hig)
        allll_to_hig.append(alll_to_hig)
        timl_to_hig.append(tim_to_hig)

    allll_ar_to_lay = np.array(allll_to_lay, dtype = object)
    allll_ar_to_hig = np.array(allll_to_hig, dtype = object)
    
    return(timl_to_lay,timl_to_hig)



def desorbing(gas_ads,net_coors,start,fin,seg_class):

    regs_allll_to_lay = []
    regs_timl_to_lay = []

    regs_allll_to_lay_prin_hig = []
    regs_timl_to_lay_prin_hig = []

    regs_allll_to_hig = []
    regs_timl_to_hig = []

    #print("pop")
    net_coors =   [[str(ele) for ele in net_coor] for net_coor in net_coors] 
    res2 = [[str(ele) for ele in gas_surf] for gas_surf in gas_ads]

    for net_coor in net_coors:
        
        al = net_coor
        allll_to_lay = []
        timl_to_lay = []
        allll_to_hig = []
        timl_to_hig = []
        numl_to_hig = []
        numl_to_lay = []      
 
        for en,(st,fi) in enumerate(zip(start,fin)):
            alll_to_lay = []
            tim_to_lay = []
            alll_to_hig = []
            tim_to_hig = []

            alll = []
            tim = []
            #print(en)
            for r in res2[st:fi]:
                ads = np.isin(r,al)
                add = 400 - len(ads)
                add = np.full((add,),"False")
                ads_add = np.concatenate((ads, add))
                #print(ads)
                #ads[0] == True: 
                condition = ads
                ind_con = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2]
                #print(ind_con)
                #print(ind_con.shape[0])
                if ads[0] == True: 
                   if seg_class == "netwo":
                      #print("shape",ind_con.shape[0],ind_con)
                      if ind_con.shape[0] == 1:
                         alll_to_lay.append(ads_add)
                         tim_to_lay.append(ind_con[0])
                      if ind_con.shape[0] > 1:
                         alll_to_hig.append(ads_add)
                         tim_to_hig.append(ind_con[0])
                   if seg_class == "high":
                      if ads[-1] == True:
                         alll_to_lay.append(ads_add)
                         tim_to_lay.append(np.sum(ind_con))
                         print("to_lay",ind_con[0], np.sum(ind_con))
                      if ads[-1] == False:
                         alll_to_hig.append(ads_add)
                         tim_to_hig.append(np.sum(ind_con))
                         print("not_to_lay",ind_con[0], np.sum(ind_con))

            allll_to_lay.append(alll_to_lay)
            timl_to_lay.append(tim_to_lay)
            #print(len(alll_to_lay))
            #print(len(tim_to_lay))
            #print(tim_to_lay)
            numl_to_hig.append(len(tim_to_hig))
            numl_to_lay.append(len(tim_to_lay))

            allll_to_hig.append(alll_to_hig)
            timl_to_hig.append(tim_to_hig)

        regs_allll_to_lay.append(allll_to_lay)
        regs_timl_to_lay.append(timl_to_lay) 

        regs_allll_to_hig.append(allll_to_hig)
        regs_timl_to_hig.append(timl_to_hig) 

        numll_to_hig = np.sum(numl_to_hig)
        numll_to_lay = np.sum(numl_to_lay)
    return(regs_timl_to_lay,regs_timl_to_hig,numll_to_lay,numll_to_hig)

 
def decays(desorbed):
    reg_timl = desorbed
    regs_decays = []
    for timl in reg_timl:
        decays = []
        for tim in timl:
            tim = np.array(tim)
            occ = []
            for i in range(1,401,1):
                con = np.count_nonzero(tim==i)
                occ.append(con)
            cumsum = 1 - (np.cumsum(occ) - occ)/np.sum(occ)
            decays.append(cumsum)
        regs_decays.append(decays)
    return(regs_decays)


inf_gas_surf = con_csv_highs(pd.read_csv("../inf_retini_xy_arr100_winds.csv"))
sup_gas_surf = con_csv_highs(pd.read_csv("../sup_retini_xy_arr100_winds.csv"))


################## 


def highs_off(gas_surfaces,coordinates_empty,coordinates_netwo,start,fin,a,b,frac):
    #coordinates_emp_net = np.concatenate((coordinates_empty,coordinates_netwo), axis = 1)
    #deso_net = desorbing(gas_surfaces,coordinates_emp_net,start[:],fin[:],"netwo")
    #
    #deso_net_to_lay = deso_net[0]
    #deso_net_to_hig = deso_net[1]

    #num_net_to_lay = np.array(deso_net[2]).reshape(1)
    #num_net_to_hig = np.array(deso_net[3]).reshape(1)

    #decay_net_to_lay          = decays(deso_net_to_lay)
    #decay_net_to_hig          = decays(deso_net_to_hig)
    #
    #decay_net_to_lay_ar       = np.array(decay_net_to_lay, dtype = object)
    #decay_net_to_hig_ar       = np.array(decay_net_to_hig, dtype = object) 
    #
    #shape1 = decay_net_to_lay_ar.shape[1] 
    #shape2 = decay_net_to_lay_ar.shape[2] 
    #
    #df_decay_net_to_lay          = pd.DataFrame(decay_net_to_lay_ar.reshape(shape1,shape2))
    #df_decay_net_to_hig          = pd.DataFrame(decay_net_to_hig_ar.reshape(shape1,shape2))
    #
    #df_decay_net_to_lay.to_csv(r""+str(a)+"_df_decay_net_to_lay_0.0_"+str(b)+".csv")         
    #df_decay_net_to_hig.to_csv(r""+str(a)+"_df_decay_net_to_hig_0.0_"+str(b)+".csv")          
    #
    #siz = len(fin[:])
    #bin_centers = np.arange(0,401,1)
    #
    #decay_net_to_lay  = np.hstack(decay_net_to_lay) 
    #decay_net_to_hig  = np.hstack(decay_net_to_hig) 
    #
    #decays_to_lay_sin = decay_single_new(decay_net_to_lay, start[:], fin[:], 0,siz)[1]
    #decays_to_hig_sin = decay_single_new(decay_net_to_hig, start[:], fin[:], 0,siz)[1]
    #decays_to_lay_dob = decay_double_new(decay_net_to_lay, start[:], fin[:], 0,siz)[1]
    #decays_to_hig_dob = decay_double_new(decay_net_to_hig, start[:], fin[:], 0,siz)[1]
    #
    #en_arr = np.array(0.000).reshape(1)
    #
    #decays_to_lay_sin = np.concatenate((en_arr,decays_to_lay_sin,num_net_to_lay))
    #decays_to_hig_sin = np.concatenate((en_arr,decays_to_hig_sin,num_net_to_hig))
    #decays_to_lay_dob = np.concatenate((en_arr,decays_to_lay_dob,num_net_to_lay))
    #decays_to_hig_dob = np.concatenate((en_arr,decays_to_hig_dob,num_net_to_hig))


    ###### End no highs off
    #
    #to_lay_sin = []
    #to_hig_sin = []
    #to_lay_dob = []
    #to_hig_dob = []
    #
    #
    #to_lay_sin.append(decays_to_lay_sin)
    #to_hig_sin.append(decays_to_hig_sin)
    #to_lay_dob.append(decays_to_lay_dob)
    #to_hig_dob.append(decays_to_hig_dob)
    
    for en in np.arange(0.001,frac,0.001):
        en = round(en,3)
        isExist = os.path.exists(r""+str(a)+"_add_net_"+str(en)+"_"+str(b)+".txt")
        if isExist == True:
           print(str(en))
           iter_coordinates_netwo= read_rdf_bis(r""+str(a)+"_add_net_"+str(en)+"_"+str(b)+".txt")
           coordinates_emp_net = np.concatenate((coordinates_empty,coordinates_netwo,iter_coordinates_netwo), axis = 1)
           print("read") 
           deso_net = desorbing(gas_surfaces,coordinates_emp_net,start[:],fin[:],"netwo")
    
           deso_net_to_lay = deso_net[0]
           deso_net_to_hig = deso_net[1]

           num_net_to_lay = np.array(deso_net[2]).reshape(1)
           num_net_to_hig = np.array(deso_net[3]).reshape(1)

           decay_net_to_lay          = decays(deso_net_to_lay)
           decay_net_to_hig          = decays(deso_net_to_hig)
    
           decay_net_to_lay_ar       = np.array(decay_net_to_lay, dtype = object)
           decay_net_to_hig_ar       = np.array(decay_net_to_hig, dtype = object)       
    
           shape1 = decay_net_to_lay_ar.shape[1] 
           shape2 = decay_net_to_lay_ar.shape[2] 
           
           df_decay_net_to_lay          = pd.DataFrame(decay_net_to_lay_ar.reshape(shape1,shape2))
           df_decay_net_to_hig          = pd.DataFrame(decay_net_to_hig_ar.reshape(shape1,shape2))
           
           df_decay_net_to_lay.to_csv(""+str(a)+"_df_decay_net_to_lay_"+str(en)+"_"+str(b)+".csv")         
           df_decay_net_to_hig.to_csv(""+str(a)+"_df_decay_net_to_hig_"+str(en)+"_"+str(b)+".csv")          
    
           siz = len(fin[:])
           bin_centers = np.arange(0,401,1)
    
           decay_net_to_lay  = np.hstack(decay_net_to_lay) 
           decay_net_to_hig  = np.hstack(decay_net_to_hig) 
    
           decays_to_lay_sin = decay_single_new(decay_net_to_lay, start[:], fin[:], 0,siz)[1]
           decays_to_hig_sin = decay_single_new(decay_net_to_hig, start[:], fin[:], 0,siz)[1]
           decays_to_lay_dob = decay_double_new(decay_net_to_lay, start[:], fin[:], 0,siz)[1]
           decays_to_hig_dob = decay_double_new(decay_net_to_hig, start[:], fin[:], 0,siz)[1]
    
           print(decays_to_lay_sin.shape)
           print(decays_to_hig_sin.shape)
           print(decays_to_lay_dob.shape)
           print(decays_to_hig_dob.shape)
    
            
           en_arr = np.array(en).reshape(1)
           print(en_arr.shape)
    
           decays_to_lay_sin = np.concatenate((en_arr,decays_to_lay_sin,num_net_to_lay))
           decays_to_hig_sin = np.concatenate((en_arr,decays_to_hig_sin,num_net_to_hig))
           decays_to_lay_dob = np.concatenate((en_arr,decays_to_lay_dob,num_net_to_lay))
           decays_to_hig_dob = np.concatenate((en_arr,decays_to_hig_dob,num_net_to_hig))
    
           print(decays_to_lay_sin.shape)
           print(decays_to_hig_sin.shape)
           print(decays_to_lay_dob.shape)
           print(decays_to_hig_dob.shape)
    
           to_lay_sin.append(decays_to_lay_sin)
           to_hig_sin.append(decays_to_hig_sin)      
           to_lay_dob.append(decays_to_lay_dob)
           to_hig_dob.append(decays_to_hig_dob)    
    
           print("end",str(en))
    
    
    
    feats1=['highs_off','t1','con1',"r2","num_ads"] 
    feats2=['highs_off','t1','t2','a','con1','con2','con_a',"r2","num_ads"]
    
    param_to_lay_sin = pd.DataFrame(data = to_lay_sin, columns =  feats1)
    param_to_hig_sin = pd.DataFrame(data = to_hig_sin, columns =  feats1)
    param_to_lay_dob = pd.DataFrame(data = to_lay_dob, columns =  feats2)
    param_to_hig_dob = pd.DataFrame(data = to_hig_dob, columns =  feats2)
    
    param_to_lay_sin.to_csv(""+str(a)+"_param_to_lay_sin_"+str(b)+".csv")         
    param_to_hig_sin.to_csv(""+str(a)+"_param_to_hig_sin_"+str(b)+".csv")          
    param_to_lay_dob.to_csv(""+str(a)+"_param_to_lay_dob_"+str(b)+".csv")         
    param_to_hig_dob.to_csv(""+str(a)+"_param_to_hig_dob_"+str(b)+".csv")          
        
    return()   

bin_centers = np.arange(0,401,1)
frac = 1.001
 
inf_coordinates_netwo     = read_rdf_bis("inf_coordinates_netwo0.13.14.15.txt")
inf_coordinates_empty     = read_rdf_bis("inf_coordinates_empty0.13.14.15.txt")
off = highs_off(inf_gas_surf,inf_coordinates_empty,inf_coordinates_netwo,inf_start[:],inf_fin[:],"inf","e1s4_e2s4",frac)

sup_coordinates_netwo     = read_rdf_bis("sup_coordinates_netwo0.13.14.15.txt")
sup_coordinates_empty     = read_rdf_bis("sup_coordinates_empty0.13.14.15.txt")
off = highs_off(sup_gas_surf,sup_coordinates_empty,sup_coordinates_netwo,sup_start[:],sup_fin[:],"sup","e1s4_e2s4",frac)
