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
    leng=[]

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

            else:
               z2_des.append(a2)
               zeta2_arr_d=np.array(z2_des)	   
               zn2_des.append(n2)
               n2_arr_d=np.array(zn2_des)	    

        a2_keep_x.append(zeta2_arr_x)
        a2_keep_y.append(zeta2_arr_y)
        a2_keep_z.append(zeta2_arr_z)

        n2_keep.append(n2_arr)
        leng.append(len(n2_arr))


        a2_des.append(zeta2_arr_d)
        n2_des.append(n2_arr_d)


    atom2_z_keep_x=np.array(a2_keep_x,dtype=object)	   
    atom2_z_keep_y=np.array(a2_keep_y,dtype=object)	   
    atom2_z_keep_z=np.array(a2_keep_z,dtype=object)	   

    num_a2_z_keep=np.array(n2_keep,dtype=object)	    


    atom2_z_des=np.array(a2_des,dtype=object)	   
    num_a2_z_des=np.array(n2_des,dtype=object)	    

    return(num_a2_z_keep,atom2_z_keep_x,atom2_z_keep_y,atom2_z_keep_z,num_a2_z_des,atom2_z_des,leng)


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
    leng = []
    inter = []
    start = 0   #   start =   10 #  10   #   10   # 10     # 10      #10     # 10   # 0
    fin   = 400  #   fin   =  250 #  250  #  250   # 80     # 230     #30     # 410  # 200
    end   = 9200 #   end   =  730 #  9610 #  730   # 20     # 9690   #970    # 9530 # 300
    final = 9600 #   final =  970 #  9850 #  970   # 90     # 9910   #990    # 9930 # 500
    step  = 400  #   step  =  240 #  240  #  240   # 2  # 5 # 220   #2      # 40   # 20
    
    for fr in range(end):
        inter.append(np.array(coord_splitted[start:fin])) 
        leng.append(np.array(coord_splitted[start]))
        start = start + step 
        fin   = fin   + step 
        if start == end:
           break   
        inter_arr = np.array(inter)
        leng_arr  = np.array(leng)
    return(inter_arr,leng_arr)

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
inf_cutoff_up_surf  =   15.69 +  translate_x_not_centered_box  #  
inf_cutoff_low_surf =   -5.0 +  translate_x_not_centered_box # -7.0 

sup_cutoff_up_surf  =    5.0  +  translate_x_not_centered_box #  6.4
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

inf_mol_x_wind = inf_all_co2_ads[6]
len_inf_inter_n_co2_ads = windowing(inf_mol_x_wind)[1]
np.savetxt('inf_co2_ads_10.txt' ,len_inf_inter_n_co2_ads, fmt = "%s"  )





sup_all_co2_ads = n_co2_ads_des(num_c,carbon,carbon[:,:,2:], sup_cutoff_up_surf, sup_cutoff_low_surf)
sup_n_co2_ads = sup_all_co2_ads[0]
sup_x_co2_ads = sup_all_co2_ads[1]
sup_y_co2_ads = sup_all_co2_ads[2]
sup_z_co2_ads = sup_all_co2_ads[3]

sup_mol_x_wind = sup_all_co2_ads[6]
len_sup_inter_n_co2_ads = windowing(sup_mol_x_wind)[1]
np.savetxt('sup_co2_ads_10.txt' ,len_sup_inter_n_co2_ads, fmt = "%s"  )


inf_inter_n_co2_ads = windowing(inf_n_co2_ads)[0]
inf_x_inter_co2_ads = windowing(inf_x_co2_ads)[0]
inf_y_inter_co2_ads = windowing(inf_y_co2_ads)[0]
inf_z_inter_co2_ads = windowing(inf_z_co2_ads)[0]
sup_inter_n_co2_ads = windowing(sup_n_co2_ads)[0]
sup_x_inter_co2_ads = windowing(sup_x_co2_ads)[0]
sup_y_inter_co2_ads = windowing(sup_y_co2_ads)[0]
sup_z_inter_co2_ads = windowing(sup_z_co2_ads)[0]

inf_all_count_id   = coor_all_multi_bis(inf_inter_n_co2_ads,inf_x_inter_co2_ads)[2]
sup_all_count_id   = coor_all_multi_bis(sup_inter_n_co2_ads,sup_x_inter_co2_ads)[2]

def decay_each_winds(all_count_id,x_inter_co2_ads,y_inter_co2_ads,z_inter_co2_ads,inter_n_co2_ads):

    app_x  = []
    app_y  = []
    app_z  = []
    app_id = []
    
    for ids,x,y,z,al in zip(all_count_id,x_inter_co2_ads,y_inter_co2_ads,z_inter_co2_ads,inter_n_co2_ads):
        app3 = []
        app4 = []
        app5 = []
        app6 = []
    
        for idss,xs,ys,zs,als in zip(ids,x,y,z,al):
            als = np.hstack(als)
            xs = np.hstack(xs)
            ys = np.hstack(ys)
            zs = np.hstack(zs)
            ind = np.isin(als,idss)
            xsk  = xs[ind]
            ysk  = ys[ind]
            zsk  = zs[ind]
            alsk = als[ind]
            app3.append(xsk)
            app4.append(ysk)
            app5.append(zsk)
            app6.append(alsk)
    
        app_x.append(app3 )  # ,dtype = object ) 
        app_y.append(app4 )  # ,dtype = object ) 
        app_z.append(app5 )
        app_id.append(app6)  # ,dtype = object )

    all_count_id = np.array(app_id,dtype = object)
    all_count_x  = np.array(app_x ,dtype = object)
    all_count_y  = np.array(app_y ,dtype = object)
    all_count_z  = np.array(app_z ,dtype = object)


    return(all_count_id,all_count_x,all_count_y,all_count_z) 

inf_dec_each_wind = decay_each_winds(inf_all_count_id,inf_x_inter_co2_ads,inf_y_inter_co2_ads,inf_z_inter_co2_ads,inf_inter_n_co2_ads)
sup_dec_each_wind = decay_each_winds(sup_all_count_id,sup_x_inter_co2_ads,sup_y_inter_co2_ads,sup_z_inter_co2_ads,sup_inter_n_co2_ads)

inf_i_dec_each_wind = inf_dec_each_wind[0]
inf_x_dec_each_wind = inf_dec_each_wind[1]
inf_y_dec_each_wind = inf_dec_each_wind[2]
inf_z_dec_each_wind = inf_dec_each_wind[3]

sup_i_dec_each_wind = sup_dec_each_wind[0]
sup_x_dec_each_wind = sup_dec_each_wind[1]
sup_y_dec_each_wind = sup_dec_each_wind[2]
sup_z_dec_each_wind = sup_dec_each_wind[3]


step  = 400
span  = step + 1
step1 = 1
time  = np.arange(1,401,step1)
np.savetxt('time.txt',time)



def group_by_ids(all_count_id,all_count_x,all_count_y,all_count_z):

    news2 = []
    x_news2 = []
    y_news2 = []
    z_news2 = []
    
    for n,x,y,z in zip(all_count_id,all_count_x,all_count_y,all_count_z):
        news = []
        x_news = []
        y_news = []
        z_news = []
        for ni in n[0]:
            n = np.hstack(n)
            x = np.hstack(x)
            y = np.hstack(y)
            z = np.hstack(z)
    
            index = np.where(n == ni)
            n_new = n[index]
            x_new = x[index]
            y_new = y[index]
            z_new = z[index]
    
            news.append(n_new)
            x_news.append(x_new)
            y_news.append(y_new)
            z_news.append(z_new)
    
            news_arr   = np.array(news  , dtype = object)
            x_news_arr = np.array(x_news, dtype = object)
            y_news_arr = np.array(y_news, dtype = object)
            z_news_arr = np.array(z_news, dtype = object)
    
        news2.append(news_arr)
        x_news2.append(x_news_arr)
        y_news2.append(y_news_arr)
        z_news2.append(z_news_arr)
    
    news_ar     = np.array(news2,dtype=object) 
    x_news_ar   = np.array(x_news2,dtype=object)
    y_news_ar   = np.array(y_news2,dtype=object)
    z_news_ar   = np.array(z_news2,dtype=object)
    
    st_news_ar     = np.hstack(news2  ) 
    st_x_news_ar   = np.hstack(x_news2)
    st_y_news_ar   = np.hstack(y_news2)
    st_z_news_ar   = np.hstack(z_news2)

    return(st_news_ar,st_x_news_ar,st_y_news_ar,st_z_news_ar)


inf_groups_ids = group_by_ids(inf_i_dec_each_wind,inf_x_dec_each_wind,inf_y_dec_each_wind,inf_z_dec_each_wind)
sup_groups_ids = group_by_ids(sup_i_dec_each_wind,sup_x_dec_each_wind,sup_y_dec_each_wind,sup_z_dec_each_wind)


inf_st_news_ar   = inf_groups_ids[0]
inf_st_x_news_ar = inf_groups_ids[1]
inf_st_y_news_ar = inf_groups_ids[2]
inf_st_z_news_ar = inf_groups_ids[3]

sup_st_news_ar   = sup_groups_ids[0]
sup_st_x_news_ar = sup_groups_ids[1]
sup_st_y_news_ar = sup_groups_ids[2]
sup_st_z_news_ar = sup_groups_ids[3]

inf_df_id = pd.DataFrame(inf_st_news_ar  )
inf_df_x  = pd.DataFrame(inf_st_x_news_ar)
inf_df_y  = pd.DataFrame(inf_st_y_news_ar)
inf_df_z  = pd.DataFrame(inf_st_z_news_ar)

inf_df_id.to_csv('wind_inf_st_co2_ids_10.csv')  
inf_df_x.to_csv('wind_inf_st_co2_x_10.csv')  
inf_df_y.to_csv('wind_inf_st_co2_y_10.csv')  
inf_df_z.to_csv('wind_inf_st_co2_z_10.csv')  

sup_df_id = pd.DataFrame(sup_st_news_ar  )
sup_df_x  = pd.DataFrame(sup_st_x_news_ar)
sup_df_y  = pd.DataFrame(sup_st_y_news_ar)
sup_df_z  = pd.DataFrame(sup_st_z_news_ar)

sup_df_id.to_csv('wind_sup_st_co2_ids_10.csv')  
sup_df_x.to_csv('wind_sup_st_co2_x_10.csv')  
sup_df_y.to_csv('wind_sup_st_co2_y_10.csv')  
sup_df_z.to_csv('wind_sup_st_co2_z_10.csv')  


exit()
