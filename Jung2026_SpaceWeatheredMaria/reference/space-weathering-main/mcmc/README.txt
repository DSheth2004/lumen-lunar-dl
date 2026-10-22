Read-me file to run the algorithm of Lapotre et al. (2017), JGR: Planets, doi: 10.1002/2016JE005248

*****************************
Stanford, 01/31/2025
Ji-In Jung
jiinjung@stanford.edu
*****************************

This version has been modified for: Jung et al. (2025), "Remote compositional analyses of space-weathered lunar maria."

- Input files: 
  (1) Spectral data: e.g. "c1lr152.txt" 
      Mare basalt spectral data example from PDS Geosciences Node Spectral Library 
      (https://pds-speclib.rsl.wustl.edu/measurement.aspx?lid=urn:nasa:pds:relab:data_reflectance:c1lr152)
      Contains a column of wavelengths (in meters) and a column of reflectance or SSA values.
  (2) Endmember optical constants: e.g. "olivine.txt" (plus 8 additional minerals)
      Contains a column of wavelengths (in meters), a column of n values, and a column of k values.

Note: The Dirichlet priors have been modified.

- load_data.m
  Loads the data. Input parameters/lines of code to adapt are highlighted in the comments.
  None of the other routines require user input.

- Main routine: MAIN.m
  This routine runs the algorithm.

- Outputs:
    (1) MAP model (displayed in the MATLAB Command Window).
    (2) A matrix containing all accepted models (grain sizes and mass abundances) saved as "Results.mat".
    (3) "BestFit_and_Residuals.jpg": A plot showing the data and the MAP model.

Note: There is an option to use SSA spectra instead of reflectance spectra; to do so, set "R1_vs_SSA0" to "0" in load_data.m.

-----------------------------
Note: This modified version will overwrite the original file(s) from Lapotre et al. (2017).
If you wish to run the original version instead, please download it from:
    https://epsp.stanford.edu/downloads
or directly via:
    https://drive.google.com/file/d/17QSbC3j323dn57jjdatoTlBnSUZmhKH5/view
-----------------------------

*****************************
Pasadena, 06/13/2017
Mathieu Lapotre
mlapotre@caltech.edu
*****************************


- Input files: 
(1) spectral data, here "O33E33A33.txt" (lab mixture of 33% olivine, 33% enstatite, 33% anorthite, D = 45-75 microns). Contains a column of wavelengths in meters, and a column of reflectance or SSA values.
(2) endmember optical constants, here "olivine.txt", etc. Contain a column of wavelengths (in meters), a column of n values, and a column of k values.

- load_data.m
Loads the data. Input parameters/lines of code to adapt are highlighted in comments. None of the other routines require user input.

- Main routine: MAIN.m
This routine runs the algorithm. 

- Outputs: 
	(1) MAP model (displayed in matlab Command Window), 
	(2) A matrix containing all accepted models (grain sizes and mass abundances) as "Results.mat", 
	(3)"BestFit_and_Residuals.jpg": Shows the data and MAP model.
	(4)"Distributions_and_Correlations.jpg": Shows distributions of accepted model parameters and their correlations. In the distribution plots, solid vertical lines indicate the MAP model, dashed lines the 68% confidence interval, and dotted lines the 95% confidence interval.

Note: There is an option to use SSA spectra instead of reflectance spectra; to do so, set "R1_vs_SSA0" to "0" in load_data.m. 


*****************************
Please cite:
Lapotre, M.G.A., B.L. Ehlmann, and S.E. Minson (2017), A probabilistic approach to remote compositional analysis of planetary surfaces, JGR: Planets, doi: 10.1002/2016JE005248.
*****************************
