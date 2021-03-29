# RetinalVesselSegmentation
  Retinal blood vessel segmentation based on U-Net. 
  The model is a U-Net, We replace Residual Double Convolution Unit with Multi-Fiber Unit, 
  which reduces the model parameters by 76.1%, while performance remains basically unchanged.
  
## Data
  Two open Dataset are used: DRIVE and CHASE_DB, and calssical preprocessing are applied: Gray-scale Conversion, Stadardization, CLAHE and γ-adjustment.
  We extract 64 * 64 pix size patches from DRIVE, 128 * 128 for CHASE_DB, then rotate patch to get 4 times more data, then normalize input data for model.
  
## Model
  Models can be found in src/U_Net2.py and src/MF_UNet2.py, proposed model is located in the latter file. 
  Multi-Fiber Unit is defined in src/MF_UNet2.py.
  It is based on grouped convolution, multiplexers are applied to ficilitate information exchange between fibers.
  ![Multi-Fiber Unit](https://github.com/mitseng/RetinalVesselSegmentation/blob/master/MFUnit.png)
  Trained model is given in /ParameterFile.

## Result
  Proposed model is trained on both dataset, classical U-Net is trained on DRIVE dataset.
  This an image from test set of DRIVE and CHASE_DB respectively.
  ![DRIVE](https://github.com/mitseng/RetinalVesselSegmentation/blob/master/DRIVE_rst.png)
  ![CHASE_DB](https://github.com/mitseng/RetinalVesselSegmentation/blob/master/CHASE_DB_rst.png)
  Various metrics are used to compare the performence of two models, the most advantage of MF-U-Net is parameter size.
  | Model    | Parameter File |   IOU   | F1-Score |
  | :------- | -------------: |  -----: | -------: |
  | U-Net    |          32.6MB|  0.6850 |   0.8131 |
  | MF-U-Net |          7.6MB |  0.6837 |   0.8121 |
