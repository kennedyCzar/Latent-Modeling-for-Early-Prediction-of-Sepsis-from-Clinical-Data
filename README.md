# Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data

Sepsis is a life-threatening condition that occurs when the body's response to infection causes tissue damage, organ failure, or death (Singer et al., 2016). In the U.S., nearly 1.7 million people develop sepsis and 270,000 people die from sepsis each year; over one third of people who die in U.S. hospitals have sepsis (CDC). Internationally, an estimated 30 million people develop sepsis and 6 million people die from sepsis each year; an estimated 4.2 million newborns and children are affected (WHO). Sepsis costs U.S. hospitals more than any other health condition at $24 billion (13% of U.S. healthcare expenses) a year, and a majority of these costs are for sepsis patients that were not diagnosed at admission (Paoli et al., 2018). Sepsis costs are even greater globally with the developing world at most risk. Altogether, sepsis is a major public health issue responsible for significant morbidity, mortality, and healthcare expenses.

Early detection and antibiotic treatment of sepsis are critical for improving sepsis outcomes, where each hour of delayed treatment has been associated with roughly an 4-8% increase in mortality (Kumar et al., 2006; Seymour et al., 2017). To help address this problem, clinicians have proposed new definitions for sepsis (Singer et al., 2016), but the fundamental need to detect and treat sepsis early still remains, and basic questions about the limits of early detection remain unanswered. The PhysioNet/Computing in Cardiology Challenge 2019 provides an opportunity to address these questions.

The early prediction of sepsis is potentially life-saving, hence, the challenge of predicting sepsis 1-hour before the clinical prediction of sepsis. Conversely, the late prediction of sepsis is potentially life-threatening, and predicting sepsis in non-sepsis patients (or predicting sepsis very early in sepsis patients) consumes limited hospital resources. For the challenge, we designed a utility function that rewards early predictions and penalizes late predictions as well as false alarms.

# Challenge Data

Data used in for this challenge is sourced from ICU patients in three separate hospital systems (source: https://physionet.org/content/challenge-2019/1.0.0/). Data from two hospital systems is publicly available; however, one is censored and used for scoring. The data for each patient will be contained within a single pipe-delimited text file. Each file will have the same header and each row will represent a single hour's worth of data. Available patient co-variates consist of Demographics, Vital Signs, and Laboratory values, which are defined in the tables below.

# How to use

- ```git clone https://github.com/kennedyCzar/Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data```
- Open and run the ```gcvae_fa.py``` file (check that you have the correct path)..Running is seemless as long as ```data.npz``` is located by script.
- Ensure to clear ```g``` folder before running the script ```gcvae_fa.py```
- Voila!
  
# Results

- The reproducable results are available in the ```g``` folder
- Simply run the script ```zclass.py``` to obtain markdown-like table and visualizations
![alt Latent](https://github.com/kennedyCzar/Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/Results/Latent_3.png)

#### Latent- $v_k \in \mathbb{R}^3$ 			                                          

|		 Model 		        |		   Factor-VAE 		    |		  MIG 		|		   Modularity 		|		 Jemmig 		|
|----------------------|--------------------------|------------|---------------------|----------------|
|	 VAE 		            |		 3387.45 +/- 23.84 		|		 0.22 		|		 0.88 		|		 0.30 		          |
|	 $\beta$-VAE 		    | 		 3379.80 +/- 24.94 	|		 0.19 		|		 0.61 		|		 0.25 		          |
|	 InfoVAE 		        |		 3378.60 +/- 26.60 		|		 0.09 		|		 0.82 		|		 0.27 		          |
|	 GCVAE $^\dagger$ 		|		 3388.10 +/- 32.26 		|		 0.10 		|		 0.89 		|		 0.43 		          |
|	 GCVAE $^\ddagger$ 	|		 3383.20 +/- 16.59 		|		 0.07 		|		 0.84 		|		 0.42 		          |



|		 Model 		|		 Total loss 		|		 Reconstruction 		|		 KL divergence |
|---------------|------------------|-----------------------|------------------|
|	 VAE 			|			 23.684 			|			 18.194 			|			 5.4896 			|
|	 $\beta$-VAE 			|			 49.568 			|			 38.921 			|			 1.0647 			|
|	 InfoVAE 			|			 22.185 			|			 18.161 			|			 15.6632 			|
|	 GCVAE $^\dagger$ 			|			 15.286 			|			 16.941 			|			 25.2697 			|
|	 GCVAE $^\ddagger$ 			|			 15.150 			|			 16.492 			|			 25.2169 			|


#### Latent- $v_k \in \mathbb{R}^{10}$

|		 Model 		|		 Factor-VAE 		|		 MIG 		|		 Modularity 		|		 Jemmig 		|
|----------------------|--------------------------|------------|---------------------|----------------|
|	 VAE 		|		 1053.20 +/- 13.72 		|		 0.21 		|		 0.77 		|		 0.28 		|
|	 $\beta$-VAE 		|		 1051.70 +/- 19.12 		|		 0.19 		|		 0.53 		|		 0.25 		|
|	 InfoVAE 		|		 1051.20 +/- 15.29 		|		 0.16 		|		 0.86 		|		 0.28 		|
|	 GCVAE $^\dagger$ 		|		 1048.80 +/- 16.37 		|		 0.11 		|		 0.92 		|		 0.29 		|
|	 GCVAE $^\ddagger$ 		|		 1050.15 +/- 14.82 		|		 0.09 		|		 0.89 		|		 0.25 		|


|		 Model 		|		 Total loss 		|		 Reconstruction 		|		 KL divergence |
|---------------|------------------|-----------------------|------------------|
|	 VAE 			|			 22.785 			|			 16.091 			|			 6.6943 			|
|	 $\beta$-VAE 			|			 49.576 			|			 38.874 			|			 1.0703 			|
|	 InfoVAE 			|			 13.743 			|			 11.967 			|			 52.1380 			|
|	 GCVAE $^\dagger$ 			|			 10.443 			|			 10.749 			|			 28.1314 			|
|	 GCVAE $^\ddagger$ 			|			 11.487 			|			 10.797 			|			 28.1542 			|

#### Latent- $v_k \in \mathbb{R}^{15}$

|		 Model 		|		 Factor-VAE 		|		 MIG 		|		 Modularity 		|		 Jemmig 		|
|----------------------|--------------------------|------------|---------------------|----------------|
|	 VAE 		|		 712.25 +/- 8.75 		|		 0.23 		|		 0.65 		|		 0.29 		|
|	 $\beta$-VAE 		|		 707.55 +/- 8.74 		|		 0.18 		|		 0.54 		|		 0.30 		|
|	 InfoVAE 		|		 711.90 +/- 12.86 		|		 0.09 		|		 0.90 		|		 0.25 		|
|	 GCVAE $^\dagger$ 		|		 714.35 +/- 12.11 		|		 0.12 		|		 0.89 		|		 0.27 		|
|	 GCVAE $^\ddagger$ 		|		 720.65 +/- 16.25 		|		 0.15 		|		 0.89 		|		 0.31 		|


|		 Model 		|		 Total loss 		|		 Reconstruction 		|		 KL divergence |
|---------------|------------------|-----------------------|------------------|
|	 VAE 			|			 22.899 			|			 16.251 			|			 6.6482 			|
|	 $\beta$-VAE 			|			 49.711 			|			 39.237 			|			 1.0474 			|
|	 InfoVAE 			|			 11.494 			|			 10.175 			|			 89.3161 			|
|	 GCVAE $^\dagger$ 			|			 9.917 			|			 9.881 			|			 28.7155 			|
|	 GCVAE $^\ddagger$ 			|			 11.313 			|			 9.672 			|			 28.9077 			|


# Paper
- Brief document explaining solution: [Paper](https://github.com/kennedyCzar/Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/Latent_Modeling_for_Early_Prediction_of_Septis.pdf)
