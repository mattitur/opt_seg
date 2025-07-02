from datetime import datetime
import sys

startTime = datetime.now()
from skimage.segmentation import watershed
from numpy import inf
import matplotlib
from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from scipy.stats import rv_continuous, gamma
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
import os, glob
import math
from functools import reduce
from numpy import linalg as la
import pdb
import matplotlib.pyplot as plt
from collections import Counter
import json


import plotly.express as px
import plotly.graph_objects as go

from scipy.special import gamma, factorial

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


from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from math import e

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

from scipy.stats import gaussian_kde
from math import e

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter

import matplotlib.ticker as mtick


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

def read_rdf(filename):
    rdf=[]
    with open(filename) as f:
         lines=f.readlines()
         for line in lines:
             rdf.append(line.strip().split("\n"))
         rdf=np.array(rdf).reshape(len(rdf))    
         rdf = rdf.astype(float)	 
         return(rdf)    

def net_emp(emp,net,last_net_emp,lim):

    empty = []
    netwo = []
    sum_empty = []
    sum_netwo = []
    
    for eall,neall,ls in zip(emp,net,last_net_emp):
     
        if (len(eall) <= len(neall)):
           cut = len(eall) + lim  
        elif (len(neall) <= len(eall)):
           cut = len(neall) + lim
    
        ne = neall[-cut:]
        e  =  eall[-cut:]
    
        su = np.insert(e, np.arange(len(ne)), ne)
        su1 = np.insert(ne, np.arange(len(e)), e)
    
        es = np.flip(np.cumsum(np.flip(su))).astype(int)
        netw = np.flip(np.cumsum(np.flip(su1))).astype(int)
    
        if (netw.all() != 0) and  (ls in netw):
           ind = int(np.where(netw == ls)[0][0])
           ind_net = (ind - 1) / 2 
           ind_net = int(ind_net)
           ind_em = (ind + 1) / 2 
           ind_em = int(ind_em)
           ne0 = ne[ind_net:]
           e0  =  e[ind_em:]
           sumen = np.sum(ne0) + np.sum(e0)
           #print("s1", sumen)
    
           if sumen == ls:
              empty.append(e0)
              netwo.append(ne0)
              sum_empty.append(np.sum(e0))
              sum_netwo.append(np.sum(ne0))
    
           if sumen != ls:
              ind_net1 = ind_net + 1
              ind_em1  =  ind_em 
              ne1 = ne[ind_net1:]
              e1  =  e[ind_em1:]
              sumen = np.sum(ne1) + np.sum(e1)
              if sumen == ls:
                 #print("s2",sumen)
                 empty.append(e1)
                 netwo.append(ne1)
                 sum_empty.append(np.sum(e1))
                 sum_netwo.append(np.sum(ne1))
    
        elif (es.all() != 0) and (ls in es):
              ind = int(np.where(es == ls)[0][0])
              ind_net = (ind + 1) / 2 
              ind_net = int(ind_net)
              ind_em = (ind - 1) / 2 
              ind_em = int(ind_em)
              ne2 = ne[ind_net:]
              e2  =  e[ind_em:]
              sumen = np.sum(ne2) + np.sum(e2)
              #print("s1", sumen)
              #print("index_es",ind,ind_net,ind_em,ne,e)
    
              if sumen == ls:
                 empty.append(e2)
                 netwo.append(ne2)
                 sum_empty.append(np.sum(e2))
                 sum_netwo.append(np.sum(ne2))
    
              if sumen != ls:
                 ind_net1 = ind_net 
                 ind_em1  =  ind_em + 1 
                 ne3 = ne[ind_net1:]
                 e3  =  e[ind_em1:]
                 sumen = np.sum(ne3) + np.sum(e3)
                 
                 if sumen == ls:
                    #print("s2",sumen)
                    empty.append(e3)
                    netwo.append(ne3)
                    sum_empty.append(np.sum(e3))
                    sum_netwo.append(np.sum(ne3))
    
        else:   
           empty.append(np.array(0).reshape(1))
           netwo.append(np.array(0).reshape(1))
           sum_empty.append(np.array(0))
           sum_netwo.append(np.array(0))

    empty = np.array(empty)
    netwo = np.array(netwo)

    return(empty,netwo,sum_empty,sum_netwo)



def combine(em1,em2):
    emp = []
    for e1,e2 in zip(em1,em2):
        #print(e1)
        if len(e1) >= len(e2):
           emp.append(e1)
        else:
           emp.append(e2)
    return(emp)



################## CV2 clustering

c  = "black"
c1 = "red"
c2 = "white"


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

def con_csv_len(df):
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
        arr = arr.astype(int)
        co2_x.append(len(arr))

    return(co2_x)

def f(x, b):
    return  np.exp(-x/b)                    #np.power(2,-x/b) ##  + b 

#def f_gamma(x, a,b):
#    return  ((1/b)**a)*(x**(a-1))*np.exp(-x/b)                    

#def f_gamma(x, a,b):
#    return ((x**(a-1))*np.exp(-x/b))/(gamma(a)*(b**a))     

#def f_gamma(x, a,b):
#    return x**(a-1)*np.exp(-x/b)/(gamma(a)*(b**a))     

def f_gamma(x, a,b):
    return x**(a-1)*np.exp(-x/b)/(gamma(a))     



#def f_gamma(x, a):
#    return (1/gamma(a))*(x**(a-1))*np.exp(-x)     



def f1(x, a):
    return a*(x)

def f2(x, a, b, c):
    return c*(np.exp(-x/a)) + (1-c)*np.exp(-x/b)

def read_files(t):
    inf_area_net = []
    sup_area_net = []
    inf_area_high = []
    sup_area_high = []


    for f in os.listdir("../"):
        if f.endswith(".txt"): 
           if f.startswith("inf_area_netwo"):
              inf_area_net.append(f)
           if f.startswith("sup_area_netwo"):
              sup_area_net.append(f)
           if f.startswith("inf_area_r"):
              inf_area_high.append(f)
           if f.startswith("sup_area_r"):
              sup_area_high.append(f)

    inf_ads_net = []
    sup_ads_net = []
    inf_ads_high = []
    sup_ads_high = []

    cwd = os.getcwd()
    print("cwd",cwd)

    for c in os.listdir("../decays/"):
        if c.endswith(".csv"): 
           if c.startswith("inf_all_ads_n"):
              inf_ads_net.append(c)
           if c.startswith("sup_all_ads_n"):
              sup_ads_net.append(c)
           if c.startswith("inf_csv_no_rt_trc_highs_split_sum"):
              inf_ads_high.append(c)
           if c.startswith("sup_csv_no_rt_trc_highs_split_sum"):
              sup_ads_high.append(c)


    return(inf_area_net,sup_area_net,inf_area_high,sup_area_high,inf_ads_net,sup_ads_net,inf_ads_high,sup_ads_high) 

def areas_sort(t,a):
    if a == "net":
       summa = []
       for filename in sorted(t, key=lambda x: (int(x.split('.')[2]) , int(x.split('.')[3]))):
           area_nets =read_rdf(filename)*0.04
           summa.append(np.sum(area_nets))
           print(filename)
    if a == "high":
       summa = []
       for filename in sorted(t, key=lambda x: (int(x.split('.')[2]) , int(x.split('.')[3]))):
           area_nets =read_rdf(filename)*0.04
           summa.append(area_nets)
           print(filename)
    return(summa)


def ads_sort(t,a):
    if a == "net":
       summa = []
       for filename in sorted(t, key=lambda x: (int(x.split('.')[2]) , int(x.split('.')[3]))):
           net =  pd.read_csv(filename)
           net =  con_csv_bis(net)
           summa.append(net)
           print("ads_net",filename)
    if a == "high":
       summa = []
       for filename in sorted(t, key=lambda x: (int(x.split('.')[2]) , int(x.split('.')[3]))):
           high =  pd.read_csv(filename)
           highs = high.to_numpy()[:,1:]*0.001

           #print(high)
           summa.append(highs)
           print("ads_high",filename)
    return(summa)



################################## inf
inf_mol_x_wind100 = read_rdf("../../85_to_100ns/inf_co2_ads_100.txt") 
inf_mol_x_wind100 = inf_mol_x_wind100.astype(int)
inf_mol_x_wind85  = read_rdf("../../70_to_85ns/inf_co2_ads_85.txt") 
inf_mol_x_wind85  = inf_mol_x_wind85.astype(int)
inf_mol_x_wind70  = read_rdf("../../55_to_70ns/inf_co2_ads_70.txt") 
inf_mol_x_wind70  = inf_mol_x_wind70.astype(int)
inf_mol_x_wind55  = read_rdf("../../40_to_55ns/inf_co2_ads_55.txt") 
inf_mol_x_wind55  = inf_mol_x_wind55.astype(int)
inf_mol_x_wind40  = read_rdf("../../25_to_40ns/inf_co2_ads_40.txt") 
inf_mol_x_wind40  = inf_mol_x_wind40.astype(int)
inf_mol_x_wind25  = read_rdf("../../10_to_25ns/inf_co2_ads_25.txt") 
inf_mol_x_wind25  = inf_mol_x_wind25.astype(int)
inf_mol_x_wind10  = read_rdf("../../0_to_10ns/inf_co2_ads_10.txt") 
inf_mol_x_wind10  = inf_mol_x_wind10.astype(int)


inf_mol_x_wind = np.concatenate((inf_mol_x_wind10,inf_mol_x_wind25,inf_mol_x_wind40,inf_mol_x_wind55,inf_mol_x_wind70,inf_mol_x_wind85,inf_mol_x_wind100),axis = 0)

inf_fin = np.cumsum(inf_mol_x_wind)
inf_start = np.append(0,inf_fin[:-1])



################################## sup

sup_mol_x_wind100 = read_rdf("../../85_to_100ns/sup_co2_ads_100.txt") 
sup_mol_x_wind100 = sup_mol_x_wind100.astype(int)
sup_mol_x_wind85  = read_rdf("../../70_to_85ns/sup_co2_ads_85.txt") 
sup_mol_x_wind85  = sup_mol_x_wind85.astype(int)
sup_mol_x_wind70  = read_rdf("../../55_to_70ns/sup_co2_ads_70.txt") 
sup_mol_x_wind70  = sup_mol_x_wind70.astype(int)
sup_mol_x_wind55  = read_rdf("../../40_to_55ns/sup_co2_ads_55.txt") 
sup_mol_x_wind55  = sup_mol_x_wind55.astype(int)
sup_mol_x_wind40  = read_rdf("../../25_to_40ns/sup_co2_ads_40.txt") 
sup_mol_x_wind40  = sup_mol_x_wind40.astype(int)
sup_mol_x_wind25  = read_rdf("../../10_to_25ns/sup_co2_ads_25.txt") 
sup_mol_x_wind25  = sup_mol_x_wind25.astype(int)
sup_mol_x_wind10  = read_rdf("../../0_to_10ns/sup_co2_ads_10.txt") 
sup_mol_x_wind10  = sup_mol_x_wind10.astype(int)


sup_mol_x_wind = np.concatenate((sup_mol_x_wind10,sup_mol_x_wind25,sup_mol_x_wind40,sup_mol_x_wind55,sup_mol_x_wind70,sup_mol_x_wind85,sup_mol_x_wind100),axis = 0)

sup_fin = np.cumsum(sup_mol_x_wind)
sup_start = np.append(0,sup_fin[:-1])

print(len(inf_fin))
print(len(sup_fin))

######### Read ads and areas data

print("tau","\u03C4")


def tot_time_highs(ads_high):
    print(ads_high)
    sum_highs_single = np.sum(ads_high, axis = 1)/(len(sup_fin)*400) # pico to nanosecond and divide by 100000 for fraction
    tot_time_highs = sum_highs_single.astype(float) 
    #tot_time_highs.append(sum_highs_single)
    return(tot_time_highs)


inf_ads_high_e1s4_e2s4 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.14.15.csv").to_numpy()[:,1:]
sup_ads_high_e1s4_e2s4 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.14.15.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s4_e2s4)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s4_e2s4)

np.savetxt("inf_tot_time_highs_e1s4_e2s4.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s4_e2s4.txt",sup_tot_time_highs   , "%s")



exit()




inf_ads_high_e1s3_e2s4 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.10.15.csv").to_numpy()[:,1:]
sup_ads_high_e1s3_e2s4 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.10.15.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s3_e2s4)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s3_e2s4)

np.savetxt("inf_tot_time_highs_e1s3_e2s4.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s3_e2s4.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s2_e2s4 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.6.15.csv").to_numpy()[:,1:]
sup_ads_high_e1s2_e2s4 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.6.15.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s2_e2s4)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s2_e2s4)

np.savetxt("inf_tot_time_highs_e1s2_e2s4.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s2_e2s4.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s1_e2s4 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.2.15.csv").to_numpy()[:,1:]
sup_ads_high_e1s1_e2s4 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.2.15.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s1_e2s4)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s1_e2s4)

np.savetxt("inf_tot_time_highs_e1s1_e2s4.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s1_e2s4.txt",sup_tot_time_highs   , "%s")


##############

inf_ads_high_e1s4_e2s3 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.14.11.csv").to_numpy()[:,1:]
sup_ads_high_e1s4_e2s3 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.14.11.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s4_e2s3)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s4_e2s3)

np.savetxt("inf_tot_time_highs_e1s4_e2s3.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s4_e2s3.txt",sup_tot_time_highs   , "%s")

inf_ads_high_e1s3_e2s3 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.10.11.csv").to_numpy()[:,1:]
sup_ads_high_e1s3_e2s3 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.10.11.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s3_e2s3)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s3_e2s3)

np.savetxt("inf_tot_time_highs_e1s3_e2s3.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s3_e2s3.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s2_e2s3 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.6.11.csv").to_numpy()[:,1:]
sup_ads_high_e1s2_e2s3 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.6.11.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s2_e2s3)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s2_e2s3)

np.savetxt("inf_tot_time_highs_e1s2_e2s3.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s2_e2s3.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s1_e2s3 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.2.11.csv").to_numpy()[:,1:]
sup_ads_high_e1s1_e2s3 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.2.11.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s1_e2s3)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s1_e2s3)

np.savetxt("inf_tot_time_highs_e1s1_e2s3.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s1_e2s3.txt",sup_tot_time_highs   , "%s")

###############

inf_ads_high_e1s4_e2s2 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.14.7.csv").to_numpy()[:,1:]
sup_ads_high_e1s4_e2s2 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.14.7.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s4_e2s2)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s4_e2s2)

np.savetxt("inf_tot_time_highs_e1s4_e2s2.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s4_e2s2.txt",sup_tot_time_highs   , "%s")

inf_ads_high_e1s3_e2s2 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.10.7.csv").to_numpy()[:,1:]
sup_ads_high_e1s3_e2s2 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.10.7.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s3_e2s2)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s3_e2s2)

np.savetxt("inf_tot_time_highs_e1s3_e2s2.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s3_e2s2.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s2_e2s2 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.6.7.csv").to_numpy()[:,1:]
sup_ads_high_e1s2_e2s2 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.6.7.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s2_e2s2)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s2_e2s2)

np.savetxt("inf_tot_time_highs_e1s2_e2s2.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s2_e2s2.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s1_e2s2 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.2.7.csv").to_numpy()[:,1:]
sup_ads_high_e1s1_e2s2 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.2.7.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s1_e2s2)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s1_e2s2)

np.savetxt("inf_tot_time_highs_e1s1_e2s2.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s1_e2s2.txt",sup_tot_time_highs   , "%s")

############

inf_ads_high_e1s4_e2s1 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.14.3.csv").to_numpy()[:,1:]
sup_ads_high_e1s4_e2s1 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.14.3.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s4_e2s1)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s4_e2s1)

np.savetxt("inf_tot_time_highs_e1s4_e2s1.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s4_e2s1.txt",sup_tot_time_highs   , "%s")

inf_ads_high_e1s3_e2s1 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.10.3.csv").to_numpy()[:,1:]
sup_ads_high_e1s3_e2s1 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.10.3.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s3_e2s1)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s3_e2s1)

np.savetxt("inf_tot_time_highs_e1s3_e2s1.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s3_e2s1.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s2_e2s1 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.6.3.csv").to_numpy()[:,1:]
sup_ads_high_e1s2_e2s1 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.6.3.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s2_e2s1)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s2_e2s1)

np.savetxt("inf_tot_time_highs_e1s2_e2s1.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s2_e2s1.txt",sup_tot_time_highs   , "%s")


inf_ads_high_e1s1_e2s1 = pd.read_csv("inf_csv_no_rt_trc_highs_split_sum_0.13.2.3.csv").to_numpy()[:,1:]
sup_ads_high_e1s1_e2s1 = pd.read_csv("sup_csv_no_rt_trc_highs_split_sum_0.13.2.3.csv").to_numpy()[:,1:]

inf_tot_time_highs = tot_time_highs(inf_ads_high_e1s1_e2s1)
sup_tot_time_highs = tot_time_highs(sup_ads_high_e1s1_e2s1)

np.savetxt("inf_tot_time_highs_e1s1_e2s1.txt",inf_tot_time_highs   , "%s")
np.savetxt("sup_tot_time_highs_e1s1_e2s1.txt",sup_tot_time_highs   , "%s")


