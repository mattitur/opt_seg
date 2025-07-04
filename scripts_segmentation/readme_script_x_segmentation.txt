Calcultion performed for the RF5, RF3 and THR segmentation group.
The python scirpts (segmentation_rf5.py, segmentation_rf3.py and segmentation_thr.py) can be run from inside each 0_100ns subfolder inside each surfaces folder (i.e. s1s5, s2s6 setc.). The scripts source the x,y positions of the adsorbed CO2 molecules from the parallel subfolders (i.e., 0_10ns, 10_25ns etc.) and outputs, among others, files needed for further analysis:

1) Segmented density maps for all different combinations of smoothing levels: "segmented_map.png".
2) Area attributed to each class for different smoothing combinations (Figure 3 in the manuscript): "area_empty.txt", "area_netwo.txt" and "area_regions_highs.txt" for LD, ID and HD, respectively.
3) Probability for each pixel to be attributed to a certain class, upon which the median value of Shannon entropy is computed (Figure 5 in the manuscript): "pixel_prob.csv".
4) Pixels coordinates for the HD regions and the merged LD + ID class (i.e. BD class) which are subsequently used to compute the residence times of individual CO2 molecules in the regions: "corr_high.csv", "coordinates_netwo_emp.txt" for HD and BD, respcetively. In the "coordinates_netwo_emp.txt" file the pixels coordinates are stored as 1d array. First and second number are x, y coordinates of first pixel and so on.   

Output files names contain numbers which refer to the level of smoothing of I, G, E1 and E2. For I number is always 0 (level of smoothing = 1), for G is always 13 (level of smoothing = 4). For E1 and E2 there are all different combinations E1 (2,6,10,14 resp. sig=1,2,3,4) and E2 (3,7,11,15 resp. sig=1,2,3,4). Prefix "inf" and "sup" refer to lower and upper surfaces, respectively. 
