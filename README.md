# Calibration-transfer-U-net

Code for Analytical Chemistry paper ‘A One-Dimensional U-Net-Based Calibration-Transfer Method for Low-Field Nuclear Magnetic Resonance Signals’ (https://doi.org/10.1021/acs.analchem.1c00765)

Please do not use this code for any commercial purposes！Violators will be prosecuted!

Citation Hou X, Wang X, Hu Y, et al. A One-Dimensional U-Net-Based Calibration-Transfer Method for Low-Field Nuclear Magnetic Resonance Signals[J]. Analytical Chemistry, 2021.

Abstract: The reconstruction of the statistical analysis model of an instrument is a time-consuming and expensive process. Herein, the feasibility of spectral model calibration-transfer application to the same type of low-field nuclear magnetic resonance (LF-NMR) instrument was investigated using a one-dimensional U-net (1D U-net). Unlike conventional calibration-transfer algorithms such as direct standardization (DS), the 1D U-net network can reduce the error between the master and slave instruments through iterative cycles. The calibration-transfer ability was verified; three experiments that entailed the use of edible oil and copper sulfate (CuSO4) samples were implemented. The analysis of the spectral responses and feature analysis of the edible oil samples revealed that the signal of the slave instrument calibrated using the 1D U-net most resembled the signal of the master instrument, and its relative residual value was reduced to 0.0045. Further analysis of the CuSO4 concentration prediction showed that on the support vector regression (SVR) model constructed using the master instrument, the signal of the slave instrument calibrated by the 1D U-net was more similar to the response of the master instrument, and its root mean square error (RMSE) was only 0.01606 mmol/L. Thus, 1D U-net is a viable candidate for calibration-transfer applications to LF-NMR instruments.

Requirements:
python 3.6.5
keras 2.2.4
tensorflow-gpu 1.11.0
