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

# Paper
- Brief document explaining solution: [Paper](https://github.com/kennedyCzar/Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/Results/Latent_3.png](https://github.com/kennedyCzar/Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/Latent_Modeling_for_Early_Prediction_of_Septis.pdf)https://github.com/kennedyCzar/Latent-Modeling-for-Early-Prediction-of-Sepsis-from-Clinical-Data/blob/main/Latent_Modeling_for_Early_Prediction_of_Septis.pdf)
