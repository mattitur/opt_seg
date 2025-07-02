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

from skimage import measure
from skimage.measure import label, regionprops
from skimage import filters

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from skimage import exposure

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


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
    # https://disq.us/p/2920ij3
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
    fin   =  200 #  200  #  200   # 80     # 220     #20     # 410  # 200
    end   =  720 #  9610 #  720   # 20     # 9690   #970    # 9520 # 200
    final =  970 #  9850 #  970   # 90     # 9910   #990    # 9920 # 500
    step  =  220 #  220  #  220   # 2  # 5 # 220   #2      # 20   # 20
    
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

def read_rdf_tris(filename):
    rdf=[]
    with open(filename) as f:
         lines=f.readlines()
         for line in lines:
             line = line.replace('[','')
             line = line.replace(']','')
             line = line.replace(' ','\n')


             rdf.append(line.strip().split("\n"))
         rdf=np.array(rdf).reshape(500,500)    
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
    co2_x = np.array(co2_x, dtype = "object")
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

########### Lower surface

def thresh(high_tau,inf_highs,a,b):

    hig = []
    low = []
    cut = []
    siz = []
    
    for n,h in enumerate(np.arange(0.001,1.001,0.001)):
    
        whig = np.where(high_tau > h)
        wlow = np.where(high_tau < h)
    
        hig.append(whig)
        low.append(wlow)
        cut.append(h)
        siz.append(whig[0].shape[0])
    
    i_siz_un, siz_un = np.flip(np.unique(siz, return_index = True))
    i_siz_un = i_siz_un.astype(int)
    cut_un = np.array(cut)[i_siz_un]

    for n,h in enumerate(np.arange(0.001,1.001,0.001)):
        if h in cut_un:
           h = np.round(h,4)
           whig = np.where(high_tau > h)
           wlow = np.where(high_tau < h)
    
           keep = inf_highs[whig]
           if keep.shape[0] > 0:
              keep_stack = np.vstack(keep)
              
              highs_keep = pd.DataFrame(keep)
              highs_keep.to_csv(r'iter_'+str(b)+'_high_'+str(h)+'_'+str(a)+'.csv') # .format(h))  

              net_bis = inf_highs[wlow]
              xc = []
              for en,c in enumerate(net_bis):
                  c = np.vstack(c)
                  xcc = []
                  for cc in c:
                      xcc.append(cc[0])  # c1 = c[:,0]  
                      xcc.append(cc[1])
              
                  xc.append(xcc)
              xc = np.hstack(xc)
              add_net = xc
              np.savetxt(r''+str(b)+'_add_net_'+str(h)+'_'+str(a)+'.txt',add_net, fmt = "%i"  )
           else:
              print("the end")
              break
  

    return()

inf_high_e1s4_e2s4 = con_csv_highs(pd.read_csv("inf_coor_high_[ 0 13 14 15].csv"))
inf_high_tau_e1s4_e2s4 = read_rdf("inf_tot_time_highs_e1s4_e2s4.txt")
thr = thresh(inf_high_tau_e1s4_e2s4,inf_high_e1s4_e2s4,"e1s4_e2s4","inf")



exit()





inf_high_e1s3_e2s4 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 10 15].csv"))
inf_high_tau_e1s3_e2s4 = read_rdf("./decays/inf_tot_time_highs_e1s3_e2s4.txt")
thr = thresh(inf_high_tau_e1s3_e2s4,inf_high_e1s3_e2s4,"e1s3_e2s4","inf")

inf_high_e1s2_e2s4 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  6 15].csv"))
inf_high_tau_e1s2_e2s4 = read_rdf("./decays/inf_tot_time_highs_e1s2_e2s4.txt")
thr = thresh(inf_high_tau_e1s2_e2s4,inf_high_e1s2_e2s4,"e1s2_e2s4","inf")

inf_high_e1s1_e2s4 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  2 15].csv"))
inf_high_tau_e1s1_e2s4 = read_rdf("./decays/inf_tot_time_highs_e1s1_e2s4.txt")
thr = thresh(inf_high_tau_e1s1_e2s4,inf_high_e1s1_e2s4,"e1s1_e2s4","inf")
#
###########
#
inf_high_e1s4_e2s3 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 14 11].csv"))
inf_high_tau_e1s4_e2s3 = read_rdf("./decays/inf_tot_time_highs_e1s4_e2s3.txt")
thr = thresh(inf_high_tau_e1s4_e2s3,inf_high_e1s4_e2s3,"e1s4_e2s3","inf")

inf_high_e1s3_e2s3 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 10 11].csv"))
inf_high_tau_e1s3_e2s3 = read_rdf("./decays/inf_tot_time_highs_e1s3_e2s3.txt")
thr = thresh(inf_high_tau_e1s3_e2s3,inf_high_e1s3_e2s3,"e1s3_e2s3","inf")

inf_high_e1s2_e2s3 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  6 11].csv"))
inf_high_tau_e1s2_e2s3 = read_rdf("./decays/inf_tot_time_highs_e1s2_e2s3.txt")
thr = thresh(inf_high_tau_e1s2_e2s3,inf_high_e1s2_e2s3,"e1s2_e2s3","inf")

inf_high_e1s1_e2s3 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  2 11].csv"))
inf_high_tau_e1s1_e2s3 = read_rdf("./decays/inf_tot_time_highs_e1s1_e2s3.txt")
thr = thresh(inf_high_tau_e1s1_e2s3,inf_high_e1s1_e2s3,"e1s1_e2s3","inf")
#
############
#
inf_high_e1s4_e2s2 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 14  7].csv"))
inf_high_tau_e1s4_e2s2 = read_rdf("./decays/inf_tot_time_highs_e1s4_e2s2.txt")
thr = thresh(inf_high_tau_e1s4_e2s2,inf_high_e1s4_e2s2,"e1s4_e2s2","inf")

inf_high_e1s3_e2s2 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 10  7].csv"))
inf_high_tau_e1s3_e2s2 = read_rdf("./decays/inf_tot_time_highs_e1s3_e2s2.txt")
thr = thresh(inf_high_tau_e1s3_e2s2,inf_high_e1s3_e2s2,"e1s3_e2s2","inf")

inf_high_e1s2_e2s2 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  6  7].csv"))
inf_high_tau_e1s2_e2s2 = read_rdf("./decays/inf_tot_time_highs_e1s2_e2s2.txt")
thr = thresh(inf_high_tau_e1s2_e2s2,inf_high_e1s2_e2s2,"e1s2_e2s2","inf")

inf_high_e1s1_e2s2 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  2  7].csv"))
inf_high_tau_e1s1_e2s2 = read_rdf("./decays/inf_tot_time_highs_e1s1_e2s2.txt")
thr = thresh(inf_high_tau_e1s1_e2s2,inf_high_e1s1_e2s2,"e1s1_e2s2","inf")
#
#############
#
inf_high_e1s4_e2s1 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 14  3].csv"))
inf_high_tau_e1s4_e2s1 = read_rdf("./decays/inf_tot_time_highs_e1s4_e2s1.txt")
thr = thresh(inf_high_tau_e1s4_e2s1,inf_high_e1s4_e2s1,"e1s4_e2s1","inf")

inf_high_e1s3_e2s1 = con_csv_highs(pd.read_csv("inf_high_[ 0 13 10  3].csv"))
inf_high_tau_e1s3_e2s1 = read_rdf("./decays/inf_tot_time_highs_e1s3_e2s1.txt")
thr = thresh(inf_high_tau_e1s3_e2s1,inf_high_e1s3_e2s1,"e1s3_e2s1","inf")

inf_high_e1s2_e2s1 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  6  3].csv"))
inf_high_tau_e1s2_e2s1 = read_rdf("./decays/inf_tot_time_highs_e1s2_e2s1.txt")
thr = thresh(inf_high_tau_e1s2_e2s1,inf_high_e1s2_e2s1,"e1s2_e2s1","inf")

inf_high_e1s1_e2s1 = con_csv_highs(pd.read_csv("inf_high_[ 0 13  2  3].csv"))
inf_high_tau_e1s1_e2s1 = read_rdf("./decays/inf_tot_time_highs_e1s1_e2s1.txt")
thr = thresh(inf_high_tau_e1s1_e2s1,inf_high_e1s1_e2s1,"e1s1_e2s1","inf")
#
##################
##################
#
sup_high_e1s4_e2s4 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 14 15].csv"))
sup_high_tau_e1s4_e2s4 = read_rdf("./decays/sup_tot_time_highs_e1s4_e2s4.txt")
thr = thresh(sup_high_tau_e1s4_e2s4,sup_high_e1s4_e2s4,"e1s4_e2s4","sup")

sup_high_e1s3_e2s4 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 10 15].csv"))
sup_high_tau_e1s3_e2s4 = read_rdf("./decays/sup_tot_time_highs_e1s3_e2s4.txt")
thr = thresh(sup_high_tau_e1s3_e2s4,sup_high_e1s3_e2s4,"e1s3_e2s4","sup")

sup_high_e1s2_e2s4 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  6 15].csv"))
sup_high_tau_e1s2_e2s4 = read_rdf("./decays/sup_tot_time_highs_e1s2_e2s4.txt")
thr = thresh(sup_high_tau_e1s2_e2s4,sup_high_e1s2_e2s4,"e1s2_e2s4","sup")

sup_high_e1s1_e2s4 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  2 15].csv"))
sup_high_tau_e1s1_e2s4 = read_rdf("./decays/sup_tot_time_highs_e1s1_e2s4.txt")
thr = thresh(sup_high_tau_e1s1_e2s4,sup_high_e1s1_e2s4,"e1s1_e2s4","sup")
#
###########
#
sup_high_e1s4_e2s3 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 14 11].csv"))
sup_high_tau_e1s4_e2s3 = read_rdf("./decays/sup_tot_time_highs_e1s4_e2s3.txt")
thr = thresh(sup_high_tau_e1s4_e2s3,sup_high_e1s4_e2s3,"e1s4_e2s3","sup")

sup_high_e1s3_e2s3 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 10 11].csv"))
sup_high_tau_e1s3_e2s3 = read_rdf("./decays/sup_tot_time_highs_e1s3_e2s3.txt")
thr = thresh(sup_high_tau_e1s3_e2s3,sup_high_e1s3_e2s3,"e1s3_e2s3","sup")

sup_high_e1s2_e2s3 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  6 11].csv"))
sup_high_tau_e1s2_e2s3 = read_rdf("./decays/sup_tot_time_highs_e1s2_e2s3.txt")
thr = thresh(sup_high_tau_e1s2_e2s3,sup_high_e1s2_e2s3,"e1s2_e2s3","sup")

sup_high_e1s1_e2s3 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  2 11].csv"))
sup_high_tau_e1s1_e2s3 = read_rdf("./decays/sup_tot_time_highs_e1s1_e2s3.txt")
thr = thresh(sup_high_tau_e1s1_e2s3,sup_high_e1s1_e2s3,"e1s1_e2s3","sup")
#
############
#
sup_high_e1s4_e2s2 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 14  7].csv"))
sup_high_tau_e1s4_e2s2 = read_rdf("./decays/sup_tot_time_highs_e1s4_e2s2.txt")
thr = thresh(sup_high_tau_e1s4_e2s2,sup_high_e1s4_e2s2,"e1s4_e2s2","sup")

sup_high_e1s3_e2s2 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 10  7].csv"))
sup_high_tau_e1s3_e2s2 = read_rdf("./decays/sup_tot_time_highs_e1s3_e2s2.txt")
thr = thresh(sup_high_tau_e1s3_e2s2,sup_high_e1s3_e2s2,"e1s3_e2s2","sup")

sup_high_e1s2_e2s2 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  6  7].csv"))
sup_high_tau_e1s2_e2s2 = read_rdf("./decays/sup_tot_time_highs_e1s2_e2s2.txt")
thr = thresh(sup_high_tau_e1s2_e2s2,sup_high_e1s2_e2s2,"e1s2_e2s2","sup")

sup_high_e1s1_e2s2 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  2  7].csv"))
sup_high_tau_e1s1_e2s2 = read_rdf("./decays/sup_tot_time_highs_e1s1_e2s2.txt")
thr = thresh(sup_high_tau_e1s1_e2s2,sup_high_e1s1_e2s2,"e1s1_e2s2","sup")
#
#############

sup_high_e1s4_e2s1 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 14  3].csv"))
sup_high_tau_e1s4_e2s1 = read_rdf("./decays/sup_tot_time_highs_e1s4_e2s1.txt")
thr = thresh(sup_high_tau_e1s4_e2s1,sup_high_e1s4_e2s1,"e1s4_e2s1","sup")

sup_high_e1s3_e2s1 = con_csv_highs(pd.read_csv("sup_high_[ 0 13 10  3].csv"))
sup_high_tau_e1s3_e2s1 = read_rdf("./decays/sup_tot_time_highs_e1s3_e2s1.txt")
thr = thresh(sup_high_tau_e1s3_e2s1,sup_high_e1s3_e2s1,"e1s3_e2s1","sup")

sup_high_e1s2_e2s1 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  6  3].csv"))
sup_high_tau_e1s2_e2s1 = read_rdf("./decays/sup_tot_time_highs_e1s2_e2s1.txt")
thr = thresh(sup_high_tau_e1s2_e2s1,sup_high_e1s2_e2s1,"e1s2_e2s1","sup")
#
sup_high_e1s1_e2s1 = con_csv_highs(pd.read_csv("sup_high_[ 0 13  2  3].csv"))
sup_high_tau_e1s1_e2s1 = read_rdf("./decays/sup_tot_time_highs_e1s1_e2s1.txt")
thr = thresh(sup_high_tau_e1s1_e2s1,sup_high_e1s1_e2s1,"e1s1_e2s1","sup")
#
