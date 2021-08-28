# XAI4EEG
XAI4EEG: Spectral and spatio-temporal explanation of Deep Learning-based Seizure Detection in EEG time series

We introduce XAI4EEG: an interactive and hybrid explanation interface for deep learning-based detection of seizures using multivariate EEG time series. The proposed interface integrates an ensemble of SHAP explainers generating local explanations, while a 1D-CNN and 3D-CNN are used to perform classification. To enhance interpretability of the explanations, we present an explanation module visualizing calculated SHAP values that enables intuitive interpretation of feature contributions in the spectral, spatial and temporal EEG dimensions.

EEG time series are characterized by (a) spectral, (b) spatial and (c) temporal dimensions. Since all three dimensions are crucial for seizure detection, we argue that an explanation of an algorithmic prediction must unify these three dimensions.
