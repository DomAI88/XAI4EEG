# XAI4EEG: Spectral and spatio-temporal explanation of Deep Learning-based Seizure Detection in EEG time series

We introduce XAI4EEG: an interactive and hybrid explanation interface for deep learning-based detection of seizures using multivariate EEG time series. The proposed interface integrates an ensemble of SHAP explainers generating local explanations, while a 1D-CNN and 3D-CNN are used to perform classification. To enhance interpretability of the explanations, we present an explanation module visualizing calculated SHAP values that enables intuitive interpretation of feature contributions in the spectral, spatial and temporal EEG dimensions.

EEG time series are characterized by (a) spectral, (b) spatial and (c) temporal dimensions. Since all three dimensions are crucial for seizure detection, we argue that an explanation of an algorithmic prediction must unify these three dimensions.

Feel free to check out our executable prototype: https://xai4eeg.herokuapp.com/

For more information: https://ml-and-vis.org/xai4eeg/

To contribute to reproducible research we offer the prototype, source code and a tutorial video.
To execute the prototype on your local machine, download the repository that incluces the following files:

* **20.zip:** 20 raw EEG intervals each with a 15-second duration
* **checkpoint_1D_CNN.h5:** 1D-CNN model artifact 
* **checkpoint_3D_CNN.h5:** 3D-CNN model artifact 
* **scaler.sav:** saved scaling parameters during data preprocessing
* **requirements.txt:** required python packages
* **X.npy:** transformed EEG data used to train 1D-CNN
* **XAI4EEG.py:** streamlit file to execute the prototype (open Anaconda Prompt and navigate to the folder where you stored the repository, finally you can launch the streamlit interface via streamlit run XAI4EEG.py)

**Please note to adjust to the code lines 79, 81, 84, 353, and 459 to your local path.**
