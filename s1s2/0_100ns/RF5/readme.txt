Calcultion performed for the RF5 segmentation group.
The python scirpt (segmentation_rf5.py) outputs, sources the x,y positions of the adsorbed CO2 molecules from the parallel subfolders (i.e., 0_10ns, 10_25ns etc.) and outputs, among others, files needed for further analysis:

1) Segmented density maps for all different combinations of smoothing levels.
2) Area attributed to each class for different smoothing combinations (Figure 3 in the manuscript).
3) Probability for each pixel to be attributed to a certain class, upon which the median value of Shannon entropy is computed (Figure 5 in the manuscript).
4) Pixels coordinates for the HD regions and the merged LD + ID class (i.e. BD class) which are subsequently used to compute the residence times of individual CO2 molecules in the regions.


