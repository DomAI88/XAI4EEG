# XAI4EEG: Spectral and spatio-temporal explanation of Deep Learning-based Seizure Detection in EEG time series

We introduce **XAI4EEG**: an application-aware approach for an explainable and hybrid deep learning-based detection of seizures in multivariate EEG time series.
In XAI4EEG, we combine deep learning models and domain knowledge on seizure detection, namely (a) frequency bands, (b) location of EEG leads and (c) temporal characteristics. From the technical perspective, XAI4EEG encompasses EEG data preparation, two deep learning models (1D-CNN and 3D-CNN) and our proposed explanation module visualizing feature contributions that are obtained by two SHAP explainers, each explaining the predictions of one of the two models.
The resulting visual explanation provides an intuitive identification of decision-relevant regions in the spectral, spatial and temporal EEG dimensions. 

EEG time series are characterized by (a) spectral, (b) spatial and (c) temporal dimensions and since all are crucial for seizure detection, *we argue that an explanation of an algorithmic prediction must unify these three dimensions.*

We provide reproducible research by offering the prototype, source code and a tutorial video:

**Executable protoype:** https://xai4eeg.herokuapp.com/
**Walkthrough video:**

To run the prototype on your local machine, please download the repository encompassing the follwing files:

* **20.zip:** 20 raw EEG intervals each with a 15-second duration
* **checkpoint_1D_CNN.h5:** 1D-CNN classifier
* **checkpoint_3D_CNN.h5:** 3D-CNN classifier
* **scaler.sav:** saved scaling parameters during our feature extraction steps
* **requirements.txt:** required packages
* **X.npy:** transformed EEG data used to train the 1D-CNN (needed to compute the SHAP values)
* **XAI4EEG.py:** streamlit main file to run the prototype; to do so, open Anaconda Prompt, navigate to the path you saved files, and run the main file via 'streamlit run XAI4EEG.py'

**Please note to adjust to the code lines 79, 81, 84, 353, and 459 in XAI4EEG.py to fit to your local path.**

For more information: https://ml-and-vis.org/xai4eeg/
