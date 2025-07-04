# opt_seg

Subfolders relative to the investigated surfaces (i.e. s1s5, s2s6 etc.) contain data used for the segmentation of the density maps and the decay analysis. The numbering of the surface follows the one in the paper (Table 2 and Table 3). Each surface folder contains in the 0_100ns (i.e. all the temporal dataframe) three subfolder relative to the different segmentation groups: RF3, RF5 and THR. Within RF3, RF5 and THR subfolder containing the data for the performed analysis are stored:

1) areas, prob and decays for RF3 and RF5.
2) areas and decays for THR.

Subfolder "scripts_segmentation" contains the python scripts to perfoem the segmentations, scipts should be run from inside the "0_100ns" subfolder which it is nested in each of the surface subfolders (i.e. s1s5, s2s6 etc.).
