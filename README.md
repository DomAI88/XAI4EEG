# XAI4EEG: Spectral and spatio-temporal explanation of Deep Learning-based Seizure Detection in EEG time series



We introduce **XAI4EEG**: an application-aware approach for an explainable and hybrid deep learning-based detection of seizures in multivariate EEG time series.
In XAI4EEG, we combine deep learning models and domain knowledge on seizure detection, namely (a) frequency bands, (b) location of EEG leads and (c) temporal characteristics. From the technical perspective, XAI4EEG encompasses EEG data preparation, two deep learning models (1D-CNN and 3D-CNN) and our proposed explanation module visualizing feature contributions that are obtained by two SHAP explainers, each explaining the predictions of one of the two models.
The resulting visual explanation provides an intuitive identification of decision-relevant regions in the spectral, spatial and temporal EEG dimensions. 

EEG time series are characterized by (a) spectral, (b) spatial and (c) temporal dimensions and since all are crucial for seizure detection, *we argue that an explanation of an algorithmic prediction must unify these three dimensions.*

We provide reproducible research by offering the prototype, a tutorial video and the source code:

**Executable protoype:** https://xai4eeg.streamlit.app/

**Tutorial video:** https://youtu.be/KHS2diEURHs

**Open Access Article:** https://link.springer.com/article/10.1007/s00521-022-07809-x

To run the prototype on your local machine, please download the repository encompassing the follwing files:

* **XAI4EEG.py:** streamlit main file to run the prototype; to do so, open Anaconda Prompt, navigate to the path you saved the files, and run the main file via 'streamlit run XAI4EEG.py'
* **20.zip:** 20 raw EEG intervals each with a 15-second duration
* **X.npy:** transformed EEG data used to train the 1D-CNN (required to compute the SHAP values)
* **scaler.sav:** encompasses the scaling parameters of our feature extraction steps
* **checkpoint_1D_CNN.h5:** 1D-CNN classifier
* **checkpoint_3D_CNN.h5:** 3D-CNN classifier
* **requirements.txt:** required packages

**Note to adjust to the code lines 79, 81, 84, 353, and 459 in XAI4EEG.py to fit your local path.**

More information: https://ml-and-vis.org/xai4eeg/
