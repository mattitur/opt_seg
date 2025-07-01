from datetime import datetime
startTime = datetime.now()
import sys
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



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import matplotlib.image as mpimg
import cv2

from itertools import groupby
from operator import itemgetter



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
    enum = []

    for en, (na2,atom2,atom2_z) in enumerate(zip(num_a,atom_type2,atom_type2_z)):
        z2_keep_x=[]
        z2_keep_y=[]
        z2_keep_z=[]
        zn2_keep=[]
        z2_des=[]
        zn2_des=[]
        enu = []
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

               enu.append(en)
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

        enum.append(enu)

    atom2_z_keep_x=np.array(a2_keep_x,dtype=object)	   
    atom2_z_keep_y=np.array(a2_keep_y,dtype=object)	   
    atom2_z_keep_z=np.array(a2_keep_z,dtype=object)	   



    num_a2_z_keep=np.array(n2_keep,dtype=object)	    


    atom2_z_des=np.array(a2_des,dtype=object)	   
    num_a2_z_des=np.array(n2_des,dtype=object)	    
    enum_arr = np.array(enum,dtype=object)
    return(num_a2_z_keep,atom2_z_keep_x,atom2_z_keep_y,atom2_z_keep_z,num_a2_z_des,atom2_z_des,enum_arr)




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
    start = 10   #   start =   10 #  10   #   10   # 10     # 10      #10     # 10   # 0
    fin   = 410  #   fin   =  250 #  250  #  250   # 80     # 230     #30     # 410  # 200
    end   = 9210 #   end   =  730 #  9610 #  730   # 20     # 9690   #970    # 9530 # 300
    final = 9610 #   final =  970 #  9850 #  970   # 90     # 9910   #990    # 9930 # 500
    step  = 400  #   step  =  240 #  240  #  240   # 2  # 5 # 220   #2      # 40   # 20
    
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


def group(n_co2_ads,x_co2_ads,y_co2_ads,z_co2_ads):
    app = []
    i_co2 = []
    x_co2 = []
    y_co2 = []
    z_co2 = []
    for ls in np.unique(c_id):
        ap  = []
        api = []
        apx = []
        apy = []
        apz = []
        for n,(ar,xc,yc,zc) in enumerate(zip(n_co2_ads,x_co2_ads,y_co2_ads,z_co2_ads)):
            if ls in ar:
               #xc = np.hstack(xc)
               #yc = np.hstack(yc)
               ind = np.where(ar == ls)
               ap.append(n)
               api.append(ar[ind])
               apx.append(xc[ind])
               apy.append(yc[ind])
               apz.append(zc[ind])
        app.append(np.hstack(ap))
        i_co2.append(np.hstack(api))
        x_co2.append(np.hstack(apx))
        y_co2.append(np.hstack(apy))
        z_co2.append(np.hstack(apz))
    app_arr   = np.array(app,dtype=object)
    i_co2_arr = np.array(i_co2,dtype=object)
    x_co2_arr = np.array(x_co2,dtype=object)
    y_co2_arr = np.array(y_co2,dtype=object)
    z_co2_arr = np.array(z_co2,dtype=object)
    return(app_arr,i_co2_arr,x_co2_arr,y_co2_arr,z_co2_arr)

def extract(app_arr,i_co2_arr,x_co2_arr,y_co2_arr,z_co2_arr):
    hop = []
    i_arr = []
    x_arr = []
    y_arr = []
    z_arr = []

    i_arr_m = []
    x_arr_m = []
    y_arr_m = []
    z_arr_m = []


    for data,i,x,y,z in zip(app_arr,i_co2_arr,x_co2_arr,y_co2_arr,z_co2_arr):
        hip = []
        i_ar = []
        x_ar = []
        y_ar = []
        z_ar = []
        for k, g in groupby(enumerate(data), lambda i_x: i_x[0] - i_x[1]):
            j = list(map(itemgetter(1), g))
            ir = np.array(i)
            xr = np.array(x)
            yr = np.array(y)
            zr = np.array(z)
            index = np.isin(data,np.array(j))
            #print("xr",xr)        
            #print("yr",yr)
            #print("index",index)
            ii = ir[index]
            xi = xr[index]
            yi = yr[index]
            zi = zr[index]
            #print(ii.shape)
            
    
            hip.append(j)
            i_ar.append(ii)
            x_ar.append(xi)
            y_ar.append(yi)
            z_ar.append(zi)

 
        i_ar = np.array(i_ar) 
        x_ar = np.array(x_ar) 
        y_ar = np.array(y_ar) 
        z_ar = np.array(z_ar) 

    
        hop.append(hip)
        i_arr.append(i_ar)
        x_arr.append(x_ar)
        y_arr.append(y_ar)
        z_arr.append(z_ar)


    i_arr = np.array(i_arr)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    z_arr = np.array(z_arr)

    for n,(ui,ux,uy,uz) in enumerate(zip(i_arr,x_arr,y_arr,z_arr)):
        if ui.ndim > 1:
           print(ui)
           print(ui.shape)
           #print(split)
           #uis = np.split(ui,[split])  # .reshape(ui.shape[1])
           #uxs = np.split(ux,[split])  # .reshape(ux.shape[1])
           #uys = np.split(uy,[split])  # .reshape(uy.shape[1])
           #uzs = np.split(uz,[split])  # .reshape(uz.shape[1])

           #i_arr_m.append(uis) 
           #x_arr_m.append(uxs) 
           #y_arr_m.append(uys) 
           #z_arr_m.append(uzs) 
        else:
           ui = ui
           ux = ux
           uy = uy
           uz = uz           

           i_arr_m.append(ui) 
           x_arr_m.append(ux) 
           y_arr_m.append(uy) 
           z_arr_m.append(uz) 



    hop_arr = np.array(hop, dtype=object)
    i_arr = np.hstack(i_arr_m)
    x_arr = np.hstack(x_arr_m)
    y_arr = np.hstack(y_arr_m)
    z_arr = np.hstack(z_arr_m)
    return(i_arr,x_arr,y_arr,z_arr)



# Set here the lammps trajectory file.

v=read_file("co2_1000fr_10ns.lammpstrj")
frames=len(v[0])

x_vectors=v[2]
y_vectors=v[3]
z_vectors=v[4]

x_vecs=box_vecs(x_vectors)[0]
y_vecs=box_vecs(y_vectors)[0]
z_vecs=box_vecs(z_vectors)[0]

x_vecs=x_vecs.reshape(len(x_vecs),1)
y_vecs=y_vecs.reshape(len(y_vecs),1)
z_vecs=z_vecs.reshape(len(z_vecs),1)
pbc_vecs = np.concatenate((x_vecs,y_vecs,z_vecs),axis=1)
pbc_vecs2 = np.concatenate((x_vecs,y_vecs),axis=1) 



recent_inf = box_vecs(z_vectors)[2]
recent_inf = recent_inf[0]
print(recent_inf)
recent_sup = box_vecs(z_vectors)[1]
recent_sup = recent_sup[0]
print(recent_sup)


coordinates=read_coord("co2_1000fr_10ns.lammpstrj",int(v[1][-1]),int(frames))

atom_numb=np.array(coordinates[0])
atom_type=np.array(coordinates[1])
atom_type=atom_type.astype(int)
atom_numb=atom_numb.astype(int)
coords=np.array(coordinates[2])
coords=coords.astype(float)


atom_n_and_coordinates=sel_atoms(atom_numb, atom_type, coords, frames,3)


num_c     = atom_n_and_coordinates[10]
carbon    = atom_n_and_coordinates[11]
num_c    = np.array(num_c) 

translate_x_not_centered_box = float(recent_sup) -  float((recent_sup - recent_inf)/2)
print(translate_x_not_centered_box)
inf_cutoff_up_surf  =   15.69 +  translate_x_not_centered_box  #  5.00
inf_cutoff_low_surf =   -6.06 +  translate_x_not_centered_box # -6.06 # -5.00

sup_cutoff_up_surf  =   6.06  +  translate_x_not_centered_box  #  5.00
sup_cutoff_low_surf =  -15.69 +  translate_x_not_centered_box # -6.06 # -5.00

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

c_id = np.unique(num_c)

inf_all_co2_ads = n_co2_ads_des(num_c,carbon,carbon[:,:,2:], inf_cutoff_up_surf, inf_cutoff_low_surf)
inf_n_co2_ads = inf_all_co2_ads[0]
inf_x_co2_ads = inf_all_co2_ads[1]
inf_y_co2_ads = inf_all_co2_ads[2]
inf_z_co2_ads = inf_all_co2_ads[3]

sup_all_co2_ads = n_co2_ads_des(num_c,carbon,carbon[:,:,2:], sup_cutoff_up_surf, sup_cutoff_low_surf)
sup_n_co2_ads = sup_all_co2_ads[0]
sup_x_co2_ads = sup_all_co2_ads[1]
sup_y_co2_ads = sup_all_co2_ads[2]
sup_z_co2_ads = sup_all_co2_ads[3]

inf_group = group(inf_n_co2_ads,inf_x_co2_ads,inf_y_co2_ads,inf_z_co2_ads)
sup_group = group(sup_n_co2_ads,sup_x_co2_ads,sup_y_co2_ads,sup_z_co2_ads)

inf_app_arr   = inf_group[0]
inf_i_co2_arr = inf_group[1]
inf_x_co2_arr = inf_group[2]
inf_y_co2_arr = inf_group[3]
inf_z_co2_arr = inf_group[4] 

sup_app_arr   = sup_group[0]
sup_i_co2_arr = sup_group[1]
sup_x_co2_arr = sup_group[2]
sup_y_co2_arr = sup_group[3]
sup_z_co2_arr = sup_group[4]

np.savetxt('inf_x_co2_arr.txt' ,inf_x_co2_arr, fmt = "%s"  )
np.savetxt('inf_y_co2_arr.txt' ,inf_y_co2_arr, fmt = "%s"  )
np.savetxt('inf_app.txt'       ,inf_app_arr, fmt = "%s"  )
np.savetxt('inf_i_co2_arr.txt' ,inf_i_co2_arr, fmt = "%s"  )


inf_extract = extract(inf_app_arr,inf_i_co2_arr,inf_x_co2_arr,inf_y_co2_arr,inf_z_co2_arr)
sup_extract = extract(sup_app_arr,sup_i_co2_arr,sup_x_co2_arr,sup_y_co2_arr,sup_z_co2_arr)

inf_i_arr = inf_extract[0] 
inf_x_arr = inf_extract[1] 
inf_y_arr = inf_extract[2] 
inf_z_arr = inf_extract[3] 

sup_i_arr = sup_extract[0] 
sup_x_arr = sup_extract[1] 
sup_y_arr = sup_extract[2] 
sup_z_arr = sup_extract[3] 

inf_df_i  = pd.DataFrame(inf_i_arr)
inf_df_x  = pd.DataFrame(inf_x_arr)
inf_df_y  = pd.DataFrame(inf_y_arr)
inf_df_z  = pd.DataFrame(inf_z_arr)

sup_df_i  = pd.DataFrame(sup_i_arr)
sup_df_x  = pd.DataFrame(sup_x_arr)
sup_df_y  = pd.DataFrame(sup_y_arr)
sup_df_z  = pd.DataFrame(sup_z_arr)


inf_df_i.to_csv('inf_st_co2_i_10ns.csv')  
inf_df_x.to_csv('inf_st_co2_x_10ns.csv')  
inf_df_y.to_csv('inf_st_co2_y_10ns.csv')  
inf_df_z.to_csv('inf_st_co2_z_10ns.csv') 

sup_df_i.to_csv('sup_st_co2_i_10ns.csv')  
sup_df_x.to_csv('sup_st_co2_x_10ns.csv')  
sup_df_y.to_csv('sup_st_co2_y_10ns.csv')  
sup_df_z.to_csv('sup_st_co2_z_10ns.csv') 


exit()

