import streamlit as st
import streamlit.components.v1 as stc
st.beta_set_page_config(
        page_title="XAI4EEG",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="collapsed",
        )

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import pandas as pd
import shap
import pickle
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import sklearn
import zipfile
import tempfile
import os
from io import StringIO
import streamlit.components.v1 as components
import mne
from mne import io, read_proj, read_selection
from mne.datasets import sample
from mne.time_frequency import psd_multitaper
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.filter import notch_filter, filter_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
pd.set_option('display.max_columns', None)
import PIL
from io import BytesIO
from PIL import Image
from pylab import *
import os
import cv2
import scipy
from scipy import signal
from scipy.integrate import simps

def main():
    bgcolor = '#FFFFFF'
    fontcolor = '#000000'
    
    html_temp = """
    
    <div style="background-color:{}; padding:10px">
    <h1 style="color:{}; text-align:center; font-size: 200%">XAI4EEG
        </h1>
    </div>

        """
    st.markdown(html_temp.format(bgcolor, fontcolor), unsafe_allow_html=True) 
        
    
    
    train_sequences_np = np.load("X.npy") 
    
    scaler=pickle.load(open("scaler.sav",'rb'))
    
    global buffer_seq
    buffer_seq = "20.zip"
    
    
    def file():
        myzipfile = zipfile.ZipFile(buffer_seq)
        with tempfile.TemporaryDirectory() as tmp_dir:
            myzipfile.extractall(tmp_dir)
            
            root_folder0 = myzipfile.namelist()[0]
            file_dir0 = os.path.join(tmp_dir, root_folder0)
            df0 = pd.read_csv(file_dir0)
            
            root_folder1 = myzipfile.namelist()[1]
            file_dir1 = os.path.join(tmp_dir, root_folder1)
            df1 = pd.read_csv(file_dir1)
            
            root_folder2 = myzipfile.namelist()[2]
            file_dir2 = os.path.join(tmp_dir, root_folder2)
            df2 = pd.read_csv(file_dir2)
            
            root_folder3 = myzipfile.namelist()[3]
            file_dir3 = os.path.join(tmp_dir, root_folder3)
            df3 = pd.read_csv(file_dir3)
            
            root_folder4 = myzipfile.namelist()[4]
            file_dir4 = os.path.join(tmp_dir, root_folder4)
            df4 = pd.read_csv(file_dir4)
            
            root_folder5 = myzipfile.namelist()[5]
            file_dir5 = os.path.join(tmp_dir, root_folder5)
            df5 = pd.read_csv(file_dir5)
            
            root_folder6 = myzipfile.namelist()[6]
            file_dir6 = os.path.join(tmp_dir, root_folder6)
            df6 = pd.read_csv(file_dir6)
            
            root_folder7 = myzipfile.namelist()[7]
            file_dir7 = os.path.join(tmp_dir, root_folder7)
            df7 = pd.read_csv(file_dir7)
            
            root_folder8 = myzipfile.namelist()[8]
            file_dir8 = os.path.join(tmp_dir, root_folder8)
            df8 = pd.read_csv(file_dir8)
            
            root_folder9 = myzipfile.namelist()[9]
            file_dir9 = os.path.join(tmp_dir, root_folder9)
            df9 = pd.read_csv(file_dir9)
            
            root_folder10 = myzipfile.namelist()[10]
            file_dir10 = os.path.join(tmp_dir, root_folder10)
            df10 = pd.read_csv(file_dir10)
            
            root_folder11 = myzipfile.namelist()[11]
            file_dir11 = os.path.join(tmp_dir, root_folder11)
            df11 = pd.read_csv(file_dir11)
            
            root_folder12 = myzipfile.namelist()[12]
            file_dir12 = os.path.join(tmp_dir, root_folder12)
            df12 = pd.read_csv(file_dir12)
            
            root_folder13 = myzipfile.namelist()[13]
            file_dir13 = os.path.join(tmp_dir, root_folder13)
            df13 = pd.read_csv(file_dir13)
            
            root_folder14 = myzipfile.namelist()[14]
            file_dir14 = os.path.join(tmp_dir, root_folder14)
            df14 = pd.read_csv(file_dir14)
            
            root_folder15 = myzipfile.namelist()[15]
            file_dir15 = os.path.join(tmp_dir, root_folder15)
            df15 = pd.read_csv(file_dir15)
            
            root_folder16 = myzipfile.namelist()[16]
            file_dir16 = os.path.join(tmp_dir, root_folder16)
            df16 = pd.read_csv(file_dir16)
            
            root_folder17 = myzipfile.namelist()[17]
            file_dir17 = os.path.join(tmp_dir, root_folder17)
            df17 = pd.read_csv(file_dir17)
            
            root_folder18 = myzipfile.namelist()[18]
            file_dir18 = os.path.join(tmp_dir, root_folder18)
            df18 = pd.read_csv(file_dir18)
            
            root_folder19 = myzipfile.namelist()[19]
            file_dir19 = os.path.join(tmp_dir, root_folder19)
            df19 = pd.read_csv(file_dir19)
            
                        
        return [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
        
    global test_sequence_df
    if buffer_seq is not None:
        df0,df1,df2,df3,df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19  = file()
        sequence_selectbox = st.selectbox("Choose interval", ("Interval 1", "Interval 2", "Interval 3", "Interval 4", "Interval 5",
                                                              "Interval 6", "Interval 7", "Interval 8", "Interval 9", "Interval 10",
                                                              "Interval 11", "Interval 12", "Interval 13", "Interval 14", "Interval 15",
                                                              "Interval 16", "Interval 17", "Interval 18", "Interval 19", "Interval 20"),
            index=0)
        

        if sequence_selectbox == "Interval 1":
            test_sequence_df = df0
        elif sequence_selectbox == "Interval 2":
            test_sequence_df = df1
        elif sequence_selectbox == "Interval 3":
            test_sequence_df = df2
        elif sequence_selectbox == "Interval 4":
            test_sequence_df = df3
        elif sequence_selectbox == "Interval 5":
            test_sequence_df = df4
        elif sequence_selectbox == "Interval 6":
            test_sequence_df = df5
        elif sequence_selectbox == "Interval 7":
            test_sequence_df = df6
        elif sequence_selectbox == "Interval 8":
            test_sequence_df = df7
        elif sequence_selectbox == "Interval 9":
            test_sequence_df = df8
        elif sequence_selectbox == "Interval 10":
            test_sequence_df = df9
        elif sequence_selectbox == "Interval 11":
            test_sequence_df = df10
        elif sequence_selectbox == "Interval 12":
            test_sequence_df = df11
        elif sequence_selectbox == "Interval 13":
            test_sequence_df = df12
        elif sequence_selectbox == "Interval 14":
            test_sequence_df = df13
        elif sequence_selectbox == "Interval 15":
            test_sequence_df = df14
        elif sequence_selectbox == "Interval 16":
            test_sequence_df = df15
        elif sequence_selectbox == "Interval 17":
            test_sequence_df = df16
        elif sequence_selectbox == "Interval 18":
            test_sequence_df = df17
        elif sequence_selectbox == "Interval 19":
            test_sequence_df = df18
        elif sequence_selectbox == "Interval 20":
            test_sequence_df = df19
        
   
         
        if test_sequence_df is not None:      
    
            #Plot Raw EEG time series
            test_sequence_df = test_sequence_df.drop(["Unnamed: 0"], axis=1)
            channel_names = test_sequence_df.columns
            seizure_loaded = test_sequence_df
            seizure_loaded_df = seizure_loaded
            seizure_loaded = seizure_loaded.values
            times = np.arange(len(seizure_loaded))
            out = [x[4:] for x in channel_names]    
            
            sf = 256
            win = None
            
            #Preprocessing Pipeline 1
            def split_dataframe(df, chunk_size = 256): 
                global chunks
                chunks = list()
                global num_chunks
                num_chunks = len(df) // chunk_size 
                for i in range(num_chunks):
                    chunks.append(df[i*chunk_size:(i+1)*chunk_size])
                return chunks
            
            split_dataframe(seizure_loaded_df, chunk_size = 256)
            seizure_loaded_df_1 = seizure_loaded_df.drop(["time"], axis=1)
            ch_names = []
            
            for i in seizure_loaded_df_1.columns:
                delta = i + str("_delta_power")
                theta = i + str("_theta_power")
                alpha = i + str("_alpha_power")
                ch_names.extend([delta, theta, alpha])
                del delta
                del theta
                del alpha
                
            df_power = pd.DataFrame(columns=[i for i in enumerate(ch_names)])
        
            df_power = df_power.append(pd.Series(name='Sec0-1'))
            df_power = df_power.append(pd.Series(name='Sec1-2'))
            df_power = df_power.append(pd.Series(name='Sec2-3'))
            df_power = df_power.append(pd.Series(name='Sec3-4'))
            df_power = df_power.append(pd.Series(name='Sec4-5'))
            df_power = df_power.append(pd.Series(name='Sec5-6'))
            df_power = df_power.append(pd.Series(name='Sec6-7'))
            df_power = df_power.append(pd.Series(name='Sec7-8'))
            df_power = df_power.append(pd.Series(name='Sec8-9'))
            df_power = df_power.append(pd.Series(name='Sec9-10'))
            df_power = df_power.append(pd.Series(name='Sec10-11'))
            df_power = df_power.append(pd.Series(name='Sec11-12'))
            df_power = df_power.append(pd.Series(name='Sec12-13'))
            df_power = df_power.append(pd.Series(name='Sec13-14'))
            df_power = df_power.append(pd.Series(name='Sec14-15'))
        
            low_delta, high_delta = 0.5, 3.5
            low_theta, high_theta = 3.5, 7.5
            low_alpha, high_alpha = 7.5, 12.5
            
            li = [] 
            for j in range(0,len(chunks)):
        
                for i in range(1,len(seizure_loaded_df.columns)):
            
                    freqs, psd = signal.welch(chunks[j].iloc[:, i], fs = sf, nperseg=256)
    
                    # Find intersecting values in frequency vector
                    idx_delta = np.logical_and(freqs >= low_delta, freqs <= high_delta)
                    idx_theta = np.logical_and(freqs >= low_theta, freqs <= high_theta)
                    idx_alpha = np.logical_and(freqs >= low_alpha, freqs <= high_alpha)
    
                    # Frequency resolution
                    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
                    delta_power = simps(psd[idx_delta], dx=freq_res)
                    theta_power = simps(psd[idx_theta], dx=freq_res)
                    alpha_power = simps(psd[idx_alpha], dx=freq_res)
            
                    li.append(delta_power)
                    li.append(theta_power)
                    li.append(alpha_power)
    
            
                    if i == 18:
                        df_power.iloc[j] =  li
                        li = []
            
            df_power["label"] = 0       
            df_all = df_power
            out_seq = df_all["label"].to_numpy(dtype=np.dtype)
            out_seq = out_seq.reshape((len(out_seq), 1))
            
            dataset_all = scaler.transform(df_all.iloc[:,0:54].to_numpy())
            dataset_all_1 = hstack((dataset_all, out_seq))
            
            # split a multivariate sequence into samples
            def split_sequences(sequences, n_steps):
            	X, y = list(), list()
            	for i in range(len(sequences)):
            		# find the end of this pattern
            		end_ix = i + n_steps
            		# check if we are beyond the dataset
            		if end_ix > len(sequences):
            			break
            		# gather input and output parts of the pattern
            		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            		X.append(seq_x)
            		y.append(seq_y)
            	return array(X), array(y)
            
            # choose a number of time steps
            n_steps = 15
            # convert into input/output
            X, y = split_sequences(dataset_all_1, n_steps)
            X = X[::15, :]
            y = y[::15,]
            X = np.asarray(X).astype('float32')
            y = np.asarray(y).astype('float32')
                       
            
            import tensorflow
            tensorflow.compat.v1.disable_v2_behavior()
                   
            #@st.cache(allow_output_mutation=True, suppress_st_warning=True)
            def shap_calc(train_sequences, X):
                
                upload_model = load_model('checkpoint_1D_CNN.h5')
                pred_probas = upload_model.predict(X, verbose=1, batch_size=len(X))
                global predictions_lstm
                predictions_lstm = np.digitize(pred_probas[:,0], [0.5])
                        
                with st.spinner('Classifying EEG sequence and calculating SHAP values ...'):
                    global explainer
                    global shap_values
                    explainer = shap.DeepExplainer(upload_model, train_sequences)
                    shap_values = explainer.shap_values(X)
                return [upload_model,shap_values, explainer]
             
            upload_model, shap_values, explainer = shap_calc(train_sequences_np, X)
            
            shap_values = np.array(shap_values)
            final_shap_values  = shap_values.reshape(shap_values.shape[2:])
            final_shap_values[final_shap_values < 0] = 0      
            
            def createComplexWavelet(time,freq,fwhm):
                sinepart = np.exp( 1j*2*np.pi*freq*time )
                gauspart = np.exp( (-4*np.log(2)*time**2)/(fwhm**2) )
                return sinepart*gauspart
            
            lis = []
            
            for i in range(1,len(seizure_loaded_df.columns)):
                
                seizure_loaded_channel = seizure_loaded[:,i]
                fs       = 256
                lofreq   = 0.5
                hifreq   = 12.5
                numfrex  = 48
                time     = np.arange(-fs,fs+1)/fs
    
                frex     = np.linspace(lofreq, hifreq, numfrex)
                fwhms    = np.linspace(4,1,numfrex)
    
                #initialize the wavelets matrix
                waveletfam = np.zeros((numfrex,len(time)),dtype=complex)
    
                for wi in range(numfrex):
                    waveletfam[wi,:] = createComplexWavelet(time,frex[wi],fwhms[wi])
            
                convres = np.convolve(seizure_loaded_channel,waveletfam[0,:], mode='same')
                # initialize
                tf = np.zeros((numfrex,len(times)))
    
                for wi in range(numfrex):
                    convres  = np.convolve(seizure_loaded_channel,waveletfam[wi,:],mode='same')
                    tf[wi,:] = np.abs(convres)
                
                vmax_var = np.max(tf) - np.max(tf[30:48]) 
                           
                #create image       
                plt.imshow(tf, aspect='auto',origin='lower',
                      extent=[times[0],times[-1],lofreq,hifreq]
                        ,vmax=vmax_var)
            
                plt.axis('off')
            
                buffer_ = BytesIO()
                plt.savefig(buffer_, format = "png", dpi = 300)
                image = PIL.Image.open(buffer_).convert('RGB')
                
    
                image = image.resize((128, 128), Image.ANTIALIAS)
                
                buffer_.seek(0)
                buffer_.close()
                plt.close()
                
                img_arr = np.asarray(image)
                lis.append(img_arr)
                         
            images_stacked = []
            images_stacked = np.array(images_stacked)
        
            images_stacked = np.stack((lis[0],
                                       lis[1],
                                        lis[2],
                                        lis[3],
                                        lis[4],
                                        lis[5],
                                        lis[6],
                                        lis[7],
                                        lis[8],
                                        lis[9],
                                        lis[10],
                                        lis[11],
                                        lis[12],
                                        lis[13],
                                        lis[14],
                                        lis[15],
                                        lis[16],
                                        lis[17])) 
            
            images_stacked = np.expand_dims(images_stacked, axis=0)
    
        
                    
            import tensorflow
            tensorflow.compat.v1.disable_v2_behavior()
            tensorflow.compat.v1.disable_eager_execution()
            
            
            def upload_cnn():
                upload_model_cnn = load_model("checkpoint_3D_CNN.h5")
                pred_probas_cnn = upload_model_cnn.predict_generator(images_stacked, verbose=1)
                global predictions_cnn
                predictions_cnn = np.digitize(pred_probas_cnn[:,0], [0.5])
                    
                return upload_model_cnn
             
            upload_model_cnn = upload_cnn()
            
            def map2layer(x, layer):
                feed_dict = dict(zip([upload_model_cnn.layers[0].input], x.copy()))
                return tensorflow.compat.v1.keras.backend.get_session().run(upload_model_cnn.layers[layer].input, feed_dict)
            
            def explain(x_train, sample, layer):
                to_explain = X[[sample]]
     
                e = shap.GradientExplainer(
                    (upload_model_cnn.layers[layer].input, upload_model_cnn.layers[-1].output),
                    map2layer(X, layer),
                    local_smoothing=0 # std dev of smoothing noise
                )
                
                global shap_values_3d
                shap_values_3d,indexes = e.shap_values(map2layer(to_explain, layer), ranked_outputs=1)
                shap_values_3d = np.array(shap_values_3d)
                                
            random = np.random.rand(1,18,128,128,3)
            
            X = np.stack([random,images_stacked], axis=0)
            X = X.astype('float32')
            X /= 255
            X = X.reshape(X.shape[0], 18,128,128,3)
            X = np.expand_dims(X, axis = 1)
            
              
            col0_row0, col1_row0, col2_row0  = st.beta_columns([2,4,2])
            
            with col1_row0: 
                
        
                st.markdown("<h1 style='text-align: center; color: black;'>Raw EEG data</h1>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: black;'>International 10-20 localization system</h2>", unsafe_allow_html=True)
                
                import matplotlib.gridspec as gridspec
                
                # RAW plots
                fig = plt.figure()
                
                gs = gridspec.GridSpec(6, 5)
                
                ax1 = plt.subplot(gs[0, 1])
                ax2 = plt.subplot(gs[0, 3])
                
                ax3 = plt.subplot(gs[1, 0])
                ax4 = plt.subplot(gs[1, 1])
                ax5 = plt.subplot(gs[1, 2])
                ax6 = plt.subplot(gs[1, 3])
                ax7 = plt.subplot(gs[1, 4])
                
                ax8 = plt.subplot(gs[2, 0])
                ax9 = plt.subplot(gs[2, 1])
                #ax10 = plt.subplot(gs[2, 2])
                ax11 = plt.subplot(gs[2, 3])
                ax12 = plt.subplot(gs[2, 4])
                
                ax13 = plt.subplot(gs[3, 0])
                ax14 = plt.subplot(gs[3, 1])
                ax15 = plt.subplot(gs[3, 2])
                ax16 = plt.subplot(gs[3, 3])
                ax17 = plt.subplot(gs[3, 4])
                
                ax18 = plt.subplot(gs[4, 1])
                ax19 = plt.subplot(gs[4, 3])
                
                ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, 
                             ax8, ax9, ax11, ax12, ax13, ax14,
                             ax15, ax16, ax17, ax18, ax19]
                
                for i, axis in  enumerate(ax_list):   
                    axis.plot(times, seizure_loaded[:,i+1])
                    axis.set_title(str(out[i+1]), fontsize=6.5)
                    axis.set_aspect('auto')
                    #axis.yaxis.set_visible(False)   
                    axis.title.set_position([0.5, 0.85])
                    
                    
                #plt.setp(ax1.get_yticklabels(), visible=False)
                plt.setp(ax1, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax1.get_xticklabels(), fontsize=5)
                plt.setp(ax1.set_ylim([-100, 100]))
                plt.setp(ax1.get_yticklabels(), fontsize=5)
                
                #plt.setp(ax2.get_yticklabels(), visible=False)
                plt.setp(ax2, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax2.get_xticklabels(), fontsize=5)
                plt.setp(ax2.set_ylim([-100, 100]))
                plt.setp(ax2.get_yticklabels(), fontsize=5)
                
                plt.setp(ax3.get_yticklabels(), visible=False)
                plt.setp(ax3, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax3.get_xticklabels(), fontsize=5)
                plt.setp(ax3.set_ylim([-100, 100]))
                plt.setp(ax3.get_yticklabels(), fontsize=5)
                
                plt.setp(ax4.get_yticklabels(), visible=False)
                plt.setp(ax4, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax4.get_xticklabels(), fontsize=5)
                plt.setp(ax4.set_ylim([-100, 100]))
                plt.setp(ax4.get_yticklabels(), fontsize=5)
                
                plt.setp(ax5.get_yticklabels(), visible=False)
                plt.setp(ax5, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax5.get_xticklabels(), fontsize=5)
                plt.setp(ax5.set_ylim([-100, 100]))
                plt.setp(ax5.get_yticklabels(), fontsize=5)
                
                plt.setp(ax6.get_yticklabels(), visible=False)
                plt.setp(ax6, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax6.get_xticklabels(), fontsize=5)
                plt.setp(ax6.set_ylim([-100, 100]))
                plt.setp(ax6.get_yticklabels(), fontsize=5)
                
                plt.setp(ax7.get_yticklabels(), visible=False)
                plt.setp(ax7, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax7.get_xticklabels(), fontsize=5)
                plt.setp(ax7.set_ylim([-100, 100]))
                plt.setp(ax7.get_yticklabels(), fontsize=5)
                
                plt.setp(ax8.get_yticklabels(), visible=False)
                plt.setp(ax8, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax8.get_xticklabels(), fontsize=5)
                plt.setp(ax8.set_ylim([-100, 100]))
                plt.setp(ax8.get_yticklabels(), fontsize=5)
                
                plt.setp(ax9.get_yticklabels(), visible=False)
                plt.setp(ax9, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax9.get_xticklabels(), fontsize=5)
                plt.setp(ax9.set_ylim([-100, 100]))
                plt.setp(ax9.get_yticklabels(), fontsize=5)
                
                plt.setp(ax11.get_yticklabels(), visible=False)
                plt.setp(ax11, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax11.get_xticklabels(), fontsize=5)
                plt.setp(ax11.set_ylim([-100, 100]))
                plt.setp(ax11.get_yticklabels(), fontsize=5)
                
                plt.setp(ax12.get_yticklabels(), visible=False)
                plt.setp(ax12, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax12.get_xticklabels(), fontsize=5)
                plt.setp(ax12.set_ylim([-100, 100]))
                plt.setp(ax12.get_yticklabels(), fontsize=5)
                
                plt.setp(ax13.get_yticklabels(), visible=False)
                plt.setp(ax13, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax13.get_xticklabels(), fontsize=5)
                plt.setp(ax13.set_ylim([-100, 100]))
                plt.setp(ax13.get_yticklabels(), fontsize=5)
                
                plt.setp(ax14.get_yticklabels(), visible=False)
                plt.setp(ax14, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax14.get_xticklabels(), fontsize=5)
                plt.setp(ax14.set_ylim([-100, 100]))
                plt.setp(ax14.get_yticklabels(), fontsize=5)
                
                plt.setp(ax15.get_yticklabels(), visible=False)
                plt.setp(ax15, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax15.get_xticklabels(), fontsize=5)
                plt.setp(ax15.set_ylim([-100, 100]))
                plt.setp(ax15.get_yticklabels(), fontsize=5)
                
                plt.setp(ax16.get_yticklabels(), visible=False)
                plt.setp(ax16, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax16.get_xticklabels(), fontsize=5)
                plt.setp(ax16.set_ylim([-100, 100]))
                plt.setp(ax16.get_yticklabels(), fontsize=5)
                
                plt.setp(ax17.get_yticklabels(), visible=False)
                plt.setp(ax17, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax17.get_xticklabels(), fontsize=5)
                plt.setp(ax17.set_ylim([-100, 100]))
                plt.setp(ax17.get_yticklabels(), fontsize=5)
                
                plt.setp(ax18.get_yticklabels(), visible=False)
                plt.setp(ax18, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax18.get_xticklabels(), fontsize=5)
                plt.setp(ax18.set_ylim([-100, 100]))
                plt.setp(ax18.get_yticklabels(), fontsize=5)
                
                plt.setp(ax19.get_yticklabels(), visible=False)
                plt.setp(ax19, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                plt.setp(ax19.get_xticklabels(), fontsize=5)
                plt.setp(ax19.set_ylim([-100, 100]))
                plt.setp(ax19.get_yticklabels(), fontsize=5)
                
                
                fig.subplots_adjust(wspace=0.2, hspace=0.8)
                #plt.title('International 10–20 system',fontdict = {'fontsize' : 60})
                plt.show()
                st.pyplot()
               
            with col0_row0: 
                #col0_row0.header("Input for 1D-CNN")
                st.markdown("<h1 style='text-align: center; color: black;'>Input for 1D-CNN</h1>", unsafe_allow_html=True)            
                import codecs           
                with st.beta_expander(""):
                    #col1_row1.header("Pre-Prcocessing Pipeline 1 (Power spectrum)")
            
                    df_power_select = df_power
                    df_power_select = df_power_select.drop(["label"], axis=1)
                    df_power_select.columns = ['Fp1-Ref_delta_power',
     'Fp1-Ref_theta_power',
     'Fp1-Ref_alpha_power',
     'Fp2-Ref_delta_power',
     'Fp2-Ref_theta_power',
     'Fp2-Ref_alpha_power',
     'F7-Ref_delta_power',
     'F7-Ref_theta_power',
     'F7-Ref_alpha_power',
     'F3-Ref_delta_power',
     ' F3-Ref_theta_power',
     ' F3-Ref_alpha_power',
     ' Fz-Ref_delta_power',
     ' Fz-Ref_theta_power',
     ' Fz-Ref_alpha_power',
     ' F4-Ref_delta_power',
     ' F4-Ref_theta_power',
     ' F4-Ref_alpha_power',
     ' F8-Ref_delta_power',
     ' F8-Ref_theta_power',
     ' F8-Ref_alpha_power',
     ' T3-Ref_delta_power',
     ' T3-Ref_theta_power',
     ' T3-Ref_alpha_power',
     ' C3-Ref_delta_power',
     ' C3-Ref_theta_power',
     ' C3-Ref_alpha_power',
     ' C4-Ref_delta_power',
     ' C4-Ref_theta_power',
     ' C4-Ref_alpha_power',
     ' T4-Ref_delta_power',
     ' T4-Ref_theta_power',
     ' T4-Ref_alpha_power',
     ' T5-Ref_delta_power',
     ' T5-Ref_theta_power',
     ' T5-Ref_alpha_power',
     ' P3-Ref_delta_power',
     ' P3-Ref_theta_power',
     ' P3-Ref_alpha_power',
     ' Pz-Ref_delta_power',
     ' Pz-Ref_theta_power',
     ' Pz-Ref_alpha_power',
     ' P4-Ref_delta_power',
     ' P4-Ref_theta_power',
     ' P4-Ref_alpha_power',
     ' T6-Ref_delta_power',
     ' T6-Ref_theta_power',
     ' T6-Ref_alpha_power',
     ' O1-Ref_delta_power',
     ' O1-Ref_theta_power',
     ' O1-Ref_alpha_power',
     ' O2-Ref_delta_power',
     ' O2-Ref_theta_power',
     ' O2-Ref_alpha_power']
    
                    #pd.options.display.float_format = "{:,.2f}".format
                    st.dataframe(df_power_select.style.format("{:,.2f}"), height=1000)
    
                    stc.html("""
        <!DOCTYPE html>
    <html>
    <head>
    <style>
    	body {
    	background-color: #ffffff
    	!important;
       margin: 2px !important;
    }
    
    * {
      transition: all .2s ease;
    }
    
    .extra-info {
      display: none;
      line-height: 30px;
      font-size: 16px;
    	position: absolute;
      top: 0;
      left: 43px;
    }
    
    .info:hover .extra-info {
      display: block;
    }
    
    .info {
      font-size: 20px;
      padding-left: 5px;
      width: 20px;
      border-radius: 15px;
    }
    
    .info:hover {
      background-color: white;
      padding: 0 0 0 5px;
      width: 315px;
      text-align: left !important;
    }
    </style>
    </head>
    <body>
    <link href="https://netdna.bootstrapcdn.com/font-awesome/3.2.1/css/font-awesome.css" rel="stylesheet">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
    
    <div class="col-md-12">
        <div class="info">
          <i class="icon-info-sign"></i>
          <span class="extra-info">
            Spectral analysis with Welch's method resulting in power spectrum.
          </span>
        </div><br />
    </div>
    </body>
    </html>
        """)
                   
                #st.table(df_power_select)
            
            with col2_row0:
                st.markdown("<h1 style='text-align: center; color: black;'>Input for 3D-CNN</h1>", unsafe_allow_html=True)
                
                with st.beta_expander(""):
                
                    methods = test_sequence_df.columns
                               
                    to_watch = images_stacked[[0]]
           
                    fig = plt.figure(figsize = (5,5))
                    
                    gs = gridspec.GridSpec(6, 5)
                    
                    ax1 = plt.subplot(gs[0, 1])
                    ax2 = plt.subplot(gs[0, 3])
                    
                    ax3 = plt.subplot(gs[1, 0])
                    ax4 = plt.subplot(gs[1, 1])
                    ax5 = plt.subplot(gs[1, 2])
                    ax6 = plt.subplot(gs[1, 3])
                    ax7 = plt.subplot(gs[1, 4])
                    
                    ax8 = plt.subplot(gs[2, 0])
                    ax9 = plt.subplot(gs[2, 1])
                    #ax10 = plt.subplot(gs[2, 2])
                    ax11 = plt.subplot(gs[2, 3])
                    ax12 = plt.subplot(gs[2, 4])
                    
                    ax13 = plt.subplot(gs[3, 0])
                    ax14 = plt.subplot(gs[3, 1])
                    ax15 = plt.subplot(gs[3, 2])
                    ax16 = plt.subplot(gs[3, 3])
                    ax17 = plt.subplot(gs[3, 4])
                    
                    ax18 = plt.subplot(gs[4, 1])
                    ax19 = plt.subplot(gs[4, 3])
                    
                    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, 
                                 ax8, ax9, ax11, ax12, ax13, ax14,
                                 ax15, ax16, ax17, ax18, ax19]
                    
                    for i, axis in  enumerate(ax_list): 
                        axis.imshow(to_watch[0][i,:,:,:], interpolation='nearest')
                        axis.set_title(str(out[i+1]), fontsize=8)
                        #axis.set_aspect('auto')
                        axis.yaxis.set_visible(False)   
                        axis.xaxis.set_visible(False) 
                        axis.title.set_position([0.5, 0.9])
                        
                    fig.subplots_adjust(wspace=0.2, hspace=0.4)
                    #fig.tight_layout()
                      #plt.title('International 10–20 system',fontdict = {'fontsize' : 60})
                    plt.show()
                    st.pyplot()
                    st.text("")
                    st.text("")
                    st.text("")
                    st.text("")
                    
                    stc.html("""
        <!DOCTYPE html>
    <html>
    <head>
    <style>
    	body {
    	background-color: #ffffff
    	!important;
       margin: 2px !important;
    }
    
    * {
      transition: all .2s ease;
    }
    
    .extra-info {
      display: none;
      line-height: 30px;
      font-size: 16px;
    	position: absolute;
      top: 0;
      left: 43px;
    }
    
    .info:hover .extra-info {
      display: block;
    }
    
    .info {
      font-size: 20px;
      padding-left: 5px;
      width: 20px;
      border-radius: 15px;
    }
    
    .info:hover {
      background-color: white;
      padding: 0 0 0 5px;
      width: 315px;
      text-align: left !important;
    }
    </style>
    </head>
    <body>
    <link href="https://netdna.bootstrapcdn.com/font-awesome/3.2.1/css/font-awesome.css" rel="stylesheet">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
    
    <div class="col-md-12">
        <div class="info">
          <i class="icon-info-sign"></i>
          <span class="extra-info">
            Time-frequency analysis with complex-valued Morlet wavelets. Time-frequency maps are concatenated along the z-axis to form a 3D image maintaining spectral, spatial, and temporal EEG information.
          </span>
        </div><br />
    </div>
    </body>
    </html>
        """)
            
            col0_row5, col1_row5, col2_row5  = st.beta_columns([1,3,1])
            
            col0_row1, col1_row1, col2_row1, col3_row1  = st.beta_columns([3,5,5,3])
            with col0_row1:
                
                
                st.markdown("<h1 style='text-align: center; color: black;'>Prediction 1D-CNN</h1>", unsafe_allow_html=True)
                if predictions_lstm == 0:
                    st.markdown("<h1 style='text-align: center; color: black;'><i>Seizure detected!</i></h1>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center; color: black;'><i>Normal pattern</i></h1>", unsafe_allow_html=True)
                    
         
            with col3_row1:
                st.markdown("<h1 style='text-align: center; color: black;'>Prediction 3D-CNN</h1>", unsafe_allow_html=True)
                if predictions_cnn == 0:
                    st.markdown("<h1 style='text-align: center; color: black;'><i>Seizure detected!</i></h1>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center; color: black;'><i>Normal pattern</i></h1>", unsafe_allow_html=True)
                   
             
            with col1_row1:
                st.markdown("<h1 style='text-align: center; color: black;'>Explanation 1D-CNN</h1>", unsafe_allow_html=True)
                  
                delta = final_shap_values[:, 0::3]
                delta_sum = np.max(delta, axis=1)
                
                theta = final_shap_values[:, 1::3]
                theta_sum = np.max(theta, axis=1)
                
                alpha = final_shap_values[:, 2::3]
                alpha_sum = np.max(alpha, axis=1)
    
                all1 = np.stack([alpha_sum,theta_sum,delta_sum])
                
                from numpy import ma
                from matplotlib import cbook
                from matplotlib.colors import Normalize
                
                class MidPointNorm(Normalize):    
                    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
                        Normalize.__init__(self,vmin, vmax, clip)
                        self.midpoint = midpoint
                
                    def __call__(self, value, clip=None):
                        if clip is None:
                            clip = self.clip
                
                        result, is_scalar = self.process_value(value)
                
                        self.autoscale_None(result)
                        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
                
                        if not (vmin < midpoint < vmax):
                            raise ValueError("midpoint must be between maxvalue and minvalue.")       
                        elif vmin == vmax:
                            result.fill(0) # Or should it be all masked? Or 0.5?
                        elif vmin > vmax:
                            raise ValueError("maxvalue must be bigger than minvalue")
                        else:
                            vmin = float(vmin)
                            vmax = float(vmax)
                            if clip:
                                mask = ma.getmask(result)
                                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                                  mask=mask)
                
                            # ma division is very slow; we can take a shortcut
                            resdat = result.data
                
                            #First scale to -1 to 1 range, than to from 0 to 1.
                            resdat -= midpoint            
                            resdat[resdat>0] /= abs(vmax - midpoint)            
                            resdat[resdat<0] /= abs(vmin - midpoint)
                
                            resdat /= 2.
                            resdat += 0.5
                            result = ma.array(resdat, mask=result.mask, copy=False)                
                
                        if is_scalar:
                            result = result[0]            
                        return result
                
                    def inverse(self, value):
                        if not self.scaled():
                            raise ValueError("Not invertible until scaled")
                        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
                
                        if cbook.iterable(value):
                            val = ma.asarray(value)
                            val = 2 * (val-0.5)  
                            val[val>0]  *= abs(vmax - midpoint)
                            val[val<0] *= abs(vmin - midpoint)
                            val += midpoint
                            return val
                        else:
                            val = 2 * (value - 0.5)
                            if val < 0: 
                                return  val*abs(vmin-midpoint) + midpoint
                            else:
                                return  val*abs(vmax-midpoint) + midpoint
                
                norm = MidPointNorm(midpoint=np.min(all1))
    
                
                rows,cols = all1.shape
                
                plt.figure(figsize = (8,2))
                plt.imshow(all1, interpolation='none', 
                     extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                    #extent=[0,15,0,2],
                     cmap='bwr', vmin=0, norm = norm)
                
                
                y_values = ["delta", "theta", "alpha"]
                y_axis = np.arange(1, 4, 1)
                plt.yticks(y_axis, y_values,fontsize=10)
                
                x_values = ["0", "5", "10", "15"]
                x_axis = np.arange(1, 16, 5)
                plt.xticks(x_axis, x_values,fontsize=10)
                
                plt.show()
                stc.html("""
        <!DOCTYPE html>
    <html>
    <head>
    <style>
    	body {
    	background-color: #ffffff
    	!important;
       margin: 2px !important;
    }
    
    * {
      transition: all .2s ease;
    }
    
    .extra-info {
      display: none;
      line-height: 30px;
      font-size: 16px;
    	position: absolute;
      top: 0;
      left: 43px;
    }
    
    .info:hover .extra-info {
      display: block;
    }
    
    .info {
      font-size: 20px;
      padding-left: 5px;
      width: 20px;
      border-radius: 15px;
    }
    
    .info:hover {
      background-color: white;
      padding: 0 0 0 5px;
      width: 315px;
      text-align: left !important;
    }
    </style>
    </head>
    <body>
    <link href="https://netdna.bootstrapcdn.com/font-awesome/3.2.1/css/font-awesome.css" rel="stylesheet">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
    
    <div class="col-md-12">
        
        <div class="info">
          <i class="icon-info-sign"></i>
          <span class="extra-info">
             Post-hoc SHAP DeepExplainer
          </span>
        </div><br />
    </div>
    </body>
    </html>
        """, height=35)
                st.pyplot()
                
                
    
            with col2_row1:
                
                st.markdown("<h1 style='text-align: center; color: black;'>Explanation 3D-CNN</h1>", unsafe_allow_html=True)
                layer_names=[layer.name for layer in upload_model_cnn.layers]
                layer_conv = list(filter(lambda k: 'conv' in k, layer_names))
                
                
            
                shap_layer_idx = layer_names.index(layer_conv[-1])
                                   
                sample = 1
            
                
                for layer in range(4,5):
                    print("layer ",layer,": ",upload_model_cnn.layers[layer])
                    explain(X, sample, layer)     
                
               
                shap_values_3d[shap_values_3d < 0] = 0
                #Delta
                li_delta = []
                for i in range(0,15):
                    start = 0 + (i*2)
                    end = 2 + (i*2)
                    temp_val = np.sum(shap_values_3d[0,0,:,0:8,start:end,:])
                    li_delta.append(temp_val)
                    
                li_delta_array = np.array(li_delta)
                
                #Theta
                li_theta = []
                for i in range(0,15):
                    start = 0 + (i*2)
                    end = 2 + (i*2)
                    temp_val = np.sum(shap_values_3d[0,0,:,8:18,start:end,:])
                    li_theta.append(temp_val)
                    
                li_theta_array = np.array(li_theta)
                
                #Alpha
                li_alpha = []
                for i in range(0,15):
                    start = 0 + (i*2)
                    end = 2 + (i*2)
                    temp_val = np.sum(shap_values_3d[0,0,:,18:31,start:end,:])
                    li_alpha.append(temp_val)
                    
                li_alpha_array = np.array(li_alpha)
                
                all2 = np.stack([li_delta_array,li_theta_array, li_alpha_array])
                #all2[all2 < 0] = 0 
                
                norm = MidPointNorm(midpoint=np.min(all2))
                rows, cols = all2.shape
                
                plt.figure(figsize = (8,2))
                plt.imshow(all2, interpolation='none', 
                     extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                    #extent=[0,15,0,2],
                     cmap='bwr', vmin=0, norm=norm)
                #plt.title('Explanation 3D-CNN',fontdict = {'fontsize' : 40})
                
                y_values = ["delta", "theta", "alpha"]
                y_axis = np.arange(1, 4, 1)
                plt.yticks(y_axis, y_values,fontsize=10)
                
                x_values = ["0", "5", "10", "15"]
                x_axis = np.arange(1, 16, 5)
                plt.xticks(x_axis, x_values,fontsize=10)
                
                plt.show()
                stc.html("""
        <!DOCTYPE html>
    <html>
    <head>
    <style>
    	body {
    	background-color: #ffffff
    	!important;
       margin: 2px !important;
    }
    
    * {
      transition: all .2s ease;
    }
    
    .extra-info {
      display: none;
      line-height: 30px;
      font-size: 16px;
    	position: absolute;
      top: 0;
      left: 43px;
    }
    
    .info:hover .extra-info {
      display: block;
    }
    
    .info {
      font-size: 20px;
      padding-left: 5px;
      width: 20px;
      border-radius: 15px;
    }
    
    .info:hover {
      background-color: white;
      padding: 0 0 0 5px;
      width: 315px;
      text-align: left !important;
    }
    </style>
    </head>
    <body>
    <link href="https://netdna.bootstrapcdn.com/font-awesome/3.2.1/css/font-awesome.css" rel="stylesheet">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css">
    
    <div class="col-md-12">
        
        <div class="info">
          <i class="icon-info-sign"></i>
          <span class="extra-info">
             Post-hoc SHAP GradientExplainer
          </span>
        </div><br />
    </div>
    </body>
    </html>
        """, height=35)
                st.pyplot()
       
         
            col0_row2, col1_row2, col2_row2  = st.beta_columns([2,4,2])
            
            with col1_row2:
                st.markdown("<h1 style='text-align: center; color: black;'>Channel-wise Explanation 1D-CNN</h1>", unsafe_allow_html=True)
                #st.header("Channel-wise explanation 1D-CNN")
                with st.beta_expander(""):
                    
                    
                    
                    fig = plt.figure()
                    #figsize=(15,5)
                    gs = gridspec.GridSpec(10, 5)
                    
                    ax1 = plt.subplot(gs[0, 1])
                    ax2 = plt.subplot(gs[0, 3])
                    
                    ax3 = plt.subplot(gs[2, 0])
                    ax4 = plt.subplot(gs[2, 1])
                    ax5 = plt.subplot(gs[2, 2])
                    ax6 = plt.subplot(gs[2, 3])
                    ax7 = plt.subplot(gs[2, 4])
                    
                    ax8 = plt.subplot(gs[4, 0])
                    ax9 = plt.subplot(gs[4, 1])
                    #ax10 = plt.subplot(gs[2, 2])
                    ax11 = plt.subplot(gs[4, 3])
                    ax12 = plt.subplot(gs[4, 4])
                    
                    ax13 = plt.subplot(gs[6, 0])
                    ax14 = plt.subplot(gs[6, 1])
                    ax15 = plt.subplot(gs[6, 2])
                    ax16 = plt.subplot(gs[6, 3])
                    ax17 = plt.subplot(gs[6, 4])
                    
                    ax18 = plt.subplot(gs[8, 1])
                    ax19 = plt.subplot(gs[8, 3])
                    
                    #Explanation
                    ax20 = plt.subplot(gs[1, 1])
                    ax21 = plt.subplot(gs[1, 3])
                    
                    ax22 = plt.subplot(gs[3, 0])
                    ax23 = plt.subplot(gs[3, 1])
                    ax24 = plt.subplot(gs[3, 2])
                    ax25 = plt.subplot(gs[3, 3])
                    ax26 = plt.subplot(gs[3, 4])
                    
                    ax27 = plt.subplot(gs[5, 0])
                    ax28 = plt.subplot(gs[5, 1])
                    #ax29 = plt.subplot(gs[2, 2])
                    ax30 = plt.subplot(gs[5, 3])
                    ax31 = plt.subplot(gs[5, 4])
                    
                    ax32 = plt.subplot(gs[7, 0])
                    ax33 = plt.subplot(gs[7, 1])
                    ax34 = plt.subplot(gs[7, 2])
                    ax35 = plt.subplot(gs[7, 3])
                    ax36 = plt.subplot(gs[7, 4])
                    ax37 = plt.subplot(gs[9, 1])
                    ax38 = plt.subplot(gs[9, 3])
                    
                    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, 
                                 ax8, ax9, ax11, ax12, ax13, ax14,
                                 ax15, ax16, ax17, ax18, ax19]
                    
                    ax_list_explainer = [ax20, ax21, ax22, ax23, ax24, 
                                         ax25, ax26, ax27, ax28, ax30,
                                         ax31, ax32, ax33, ax34, ax35,
                                         ax36, ax37, ax38]
                    
          
                    for i, axis in  enumerate(ax_list):  
                        axis.plot(times, seizure_loaded[:,i+1])
                        axis.set_title(str(out[i+1]), fontsize=6.5)
                        axis.set_aspect('auto')
                        #axis.yaxis.set_visible(False)
                        #axis.xaxis.set_visible(False)
                        axis.title.set_position([0.5, 0.75])
                           
                    for i, axis in  enumerate(ax_list_explainer):
                        
                        temp = final_shap_values[:,0+(i*3):3+(i*3)]
                        temp = np.moveaxis(temp, 0,-1)
                        rows,cols = temp.shape
                        norm = MidPointNorm(midpoint=np.min(temp))
                        axis.imshow(temp, interpolation='none', cmap='bwr',
                         extent=[0.5, 0.5+cols, 0.5, 0.5+rows],
                         vmin=-0.000001,  vmax=np.max(final_shap_values), norm = norm, origin="lower")
                        #vmax=np.max(final_shap_values),
                        axis.set_aspect('auto')
                        if (i == 0) or (i==2) or (i==7) or (i==11) or (i==16):
                            axis.set_yticklabels(['delta', 'theta', 'alpha'], fontsize=6)
                            axis.set_xticklabels([])
                            axis.xaxis.set_visible(False)
                           
                        else:
                            
                            axis.yaxis.set_visible(False)
                            axis.set_xticklabels([])
                            axis.xaxis.set_visible(False)
                           
             
                    y_axis = np.arange(1, 4, 1)
                    plt.setp(ax20, yticks=y_axis, yticklabels=["delta", "theta", "alpha"])
                    plt.setp(ax20.get_yticklabels(), fontsize=5)
                    
                    y_axis = np.arange(1, 4, 1)
                    plt.setp(ax22, yticks=y_axis, yticklabels=["delta", "theta", "alpha"])
                    plt.setp(ax22.get_yticklabels(), fontsize=5)
                    
                    y_axis = np.arange(1, 4, 1)
                    plt.setp(ax27, yticks=y_axis, yticklabels=["delta", "theta", "alpha"])
                    plt.setp(ax27.get_yticklabels(), fontsize=5)
                    
                    y_axis = np.arange(1, 4, 1)
                    plt.setp(ax32, yticks=y_axis, yticklabels=["delta", "theta", "alpha"])
                    plt.setp(ax32.get_yticklabels(), fontsize=5)
                    
                    y_axis = np.arange(1, 4, 1)
                    plt.setp(ax37, yticks=y_axis, yticklabels=["delta", "theta", "alpha"])
                    plt.setp(ax37.get_yticklabels(), fontsize=5)
                               
                    #plt.setp(ax1.get_yticklabels(), visible=False)
                    plt.setp(ax1, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax1.get_xticklabels(), fontsize=5)
                    plt.setp(ax1.set_ylim([-100, 100]))
                    plt.setp(ax1.get_yticklabels(), fontsize=5)
                    
                    #plt.setp(ax2.get_yticklabels(), visible=False)
                    plt.setp(ax2, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax2.get_xticklabels(), fontsize=5)
                    plt.setp(ax2.set_ylim([-100, 100]))
                    plt.setp(ax2.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax3.get_yticklabels(), visible=False)
                    plt.setp(ax3, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax3.get_xticklabels(), fontsize=5)
                    plt.setp(ax3.set_ylim([-100, 100]))
                    plt.setp(ax3.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax4.get_yticklabels(), visible=False)
                    plt.setp(ax4, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax4.get_xticklabels(), fontsize=5)
                    plt.setp(ax4.set_ylim([-100, 100]))
                    plt.setp(ax4.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax5.get_yticklabels(), visible=False)
                    plt.setp(ax5, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax5.get_xticklabels(), fontsize=5)
                    plt.setp(ax5.set_ylim([-100, 100]))
                    plt.setp(ax5.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax6.get_yticklabels(), visible=False)
                    plt.setp(ax6, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax6.get_xticklabels(), fontsize=5)
                    plt.setp(ax6.set_ylim([-100, 100]))
                    plt.setp(ax6.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax7.get_yticklabels(), visible=False)
                    plt.setp(ax7, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax7.get_xticklabels(), fontsize=5)
                    plt.setp(ax7.set_ylim([-100, 100]))
                    plt.setp(ax7.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax8.get_yticklabels(), visible=False)
                    plt.setp(ax8, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax8.get_xticklabels(), fontsize=5)
                    plt.setp(ax8.set_ylim([-100, 100]))
                    plt.setp(ax8.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax9.get_yticklabels(), visible=False)
                    plt.setp(ax9, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax9.get_xticklabels(), fontsize=5)
                    plt.setp(ax9.set_ylim([-100, 100]))
                    plt.setp(ax9.get_yticklabels(), fontsize=5)
                    
                    
                    plt.setp(ax11.get_yticklabels(), visible=False)
                    plt.setp(ax11, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax11.get_xticklabels(), fontsize=5)
                    plt.setp(ax11.set_ylim([-100, 100]))
                    plt.setp(ax11.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax12.get_yticklabels(), visible=False)
                    plt.setp(ax12, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax12.get_xticklabels(), fontsize=5)
                    plt.setp(ax12.set_ylim([-100, 100]))
                    plt.setp(ax12.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax13.get_yticklabels(), visible=False)
                    plt.setp(ax13, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax13.get_xticklabels(), fontsize=5)
                    plt.setp(ax13.set_ylim([-100, 100]))
                    plt.setp(ax13.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax14.get_yticklabels(), visible=False)
                    plt.setp(ax14, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax14.get_xticklabels(), fontsize=5)
                    plt.setp(ax14.set_ylim([-100, 100]))
                    plt.setp(ax14.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax15.get_yticklabels(), visible=False)
                    plt.setp(ax15, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax15.get_xticklabels(), fontsize=5)
                    plt.setp(ax15.set_ylim([-100, 100]))
                    plt.setp(ax15.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax16.get_yticklabels(), visible=False)
                    plt.setp(ax16, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax16.get_xticklabels(), fontsize=5)
                    plt.setp(ax16.set_ylim([-100, 100]))
                    plt.setp(ax16.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax17.get_yticklabels(), visible=False)
                    plt.setp(ax17, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax17.get_xticklabels(), fontsize=5)
                    plt.setp(ax17.set_ylim([-100, 100]))
                    plt.setp(ax17.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax18.get_yticklabels(), visible=False)
                    plt.setp(ax18, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax18.get_xticklabels(), fontsize=5)
                    plt.setp(ax18.set_ylim([-100, 100]))
                    plt.setp(ax18.get_yticklabels(), fontsize=5)
                    
                    plt.setp(ax19.get_yticklabels(), visible=False)
                    plt.setp(ax19, xticks=[0,1280,2560,3840], xticklabels=['0', '5', '10', '15'])
                    plt.setp(ax19.get_xticklabels(), fontsize=5)
                    plt.setp(ax19.set_ylim([-100, 100]))
                    plt.setp(ax19.get_yticklabels(), fontsize=5)
                    
                    
                    fig.subplots_adjust(wspace=0.2, hspace=0.8)
                    plt.subplot_tool()
                    plt.show()
                    st.pyplot()
                
            col0_row3, col1_row3, col2_row3  = st.beta_columns([2,4,2])
            #st.info('This is a purely informational message'      
            
            with col1_row3:
                st.markdown("<h1 style='text-align: center; color: black;'>True label</h1>", unsafe_allow_html=True)
                #st.header("Channel-wise explanation 1D-CNN")
                with st.beta_expander(""):
                    if sequence_selectbox == "Interval 1":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 2":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 3":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)    
                        
                    elif sequence_selectbox == "Interval 4":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 5":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 6":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True) 
                        
                    elif sequence_selectbox == "Interval 7":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 8":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                        
                    elif sequence_selectbox == "Interval 9":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 10":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)   
                       
                    elif sequence_selectbox == "Interval 11":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)
                        
                    elif sequence_selectbox == "Interval 12":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)   
                       
                    elif sequence_selectbox == "Interval 13":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)   
                       
                    elif sequence_selectbox == "Interval 14":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                        
                    elif sequence_selectbox == "Interval 15":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                       
                    elif sequence_selectbox == "Interval 16":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                        
                    elif sequence_selectbox == "Interval 17":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                        
                    elif sequence_selectbox == "Interval 18":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                      
                    elif sequence_selectbox == "Interval 19":
                        st.markdown("<h1 style='text-align: center; color: black;'>Seizure</h1>", unsafe_allow_html=True)
                        
                    elif sequence_selectbox == "Interval 20":
                        st.markdown("<h1 style='text-align: center; color: black;'>Normal pattern</h1>", unsafe_allow_html=True)  
                        
    
                    
main()


