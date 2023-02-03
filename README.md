<p align="center">
 
 <h2 align="center">Reconstructing Twelve-Lead ECG Signals from Single-Lead Signals Using
Generative Adversarial Networks(under review)</h2>

### Subject : ECG_Reconstruction for detect multi label diseases
 
-   we focus 12 Lead ECG from Lead I
-   model name EKGAN(Elektrokardiogramm GAN)      
 <br/>
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/81897022/211444031-9ad4e7a6-7851-44ff-94b0-c49f2827f222.png" alt="text" width="number" />
</p>

### ECG domain
-   12 Lead ECG provides spatial information about heart's electrical activity in 3 approximately orthogonal directions
-   12 Lead ECG consist of Standard Limb Leads(Lead I, Lead II, Lead III), Augmented Limb Leads(aVR, aVF, aVL) and Precordial Lead(V1, V2, V3, V4, V5, V6)

-   Lead I : RA (-) to LA (+) (Right Left, or lateral)
-   Lead II: RA (-) to LL (+) (Superior Inferior)
-   Lead III: LA (-) to LL (+) (Superior Inferior)
-   Lead aVR: RA (+) to [LA & LL] (-) (Rightward)
-   Lead aVL: LA (+) to [RA & LL] (-) (Leftward)
-   Lead aVF: LL (+) to [RA & LA] (-) (Inferior)
-   Leads V1, V2, V3: (Posterior Anterior)
-   Leads V4, V5, V6:(Right Left, or lateral)
* RA : right arm, LA : left arm, LL : left foot



 <br/>
</p>
 
 ![fig-reconstruction (2)](https://user-images.githubusercontent.com/81897022/211257601-fa974428-2579-4a56-bd4d-08d9bed0dfa4.png)

</p>








<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=for-the-badge&logo=Tensorflow&logoColor=white">
   <img src="https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white">
</p>  

if you want more information, please send me an e-mail at jinho381@naver.com

</p>

# Usage

-   U-net
-   GAN
-   Signal processing
-   ECG domain knowledge


#### we evaluate another model such as

-   `pix2pix(CVPR, 2017)` - https://arxiv.org/abs/1611.07004
-   `cycleGAN(ICCV, 2017)` - https://arxiv.org/abs/1703.10593
-   `CardioGAN(AAAI, 2021)` - https://arxiv.org/abs/2010.00104
-   `EKGAN(under review)` - 







