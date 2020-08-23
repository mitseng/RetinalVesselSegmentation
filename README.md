# RetinalVesselSegmentation
  Retinal blood vessel segmentation based on FCN. 
  The modle is based on U-Net, We replace Residual Double Convolution Unit with Multi-Fiber Unit, 
  which reduces the model parameters by 76.1%, while performance remains basically unchanged.
  
## Data
  Two open Dataset are used: DRIVE and CHASE_DB, and calssical preprocessing are applied: Gray-scale Conversion, Stadardization, CLAHE and γ-adjustment.
  We extract 64 * 64 pix size patches from DRIVE, 128 * 128 for CHASE_DB, then rotate patch to get 4 times more data, then normalize input data for modle.
  
