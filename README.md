# SERENEt
SERENEt: SEgmentation and REcover NEtwork for SKA-Low multi-frequency tomographic 21-cm data for the Epoch of Reionization (EoR). The fulo SERENEt code that consist of a pre-process step, for foreground mitigation, and two U-shaped neural network for segmentation and 21-cm signal recovery, respectively. A general overview is given in Figure 1.<br/>
<img src="https://github.com/micbia/serenet/blob/main/docs/SERENEt_pipeline.png"><br/>
Figure 1: A simplified view of the project pipeline. The data include the mock observation with foreground and instrumental noise contamination. The residual image after the pre-process step. The binary prior for neutral region identification and the recovered 21-cm image. Data input and output are shown with an example image.<br/><br/>

<ul>
  <li> In the first part, before applying any machine learning method, we process the challenge input data, <i>I_obs</i> , with an algorithm that partially subtracts the foreground contamination. The resulting residual image will still contain some foreground residual and most systematic noise. However, this pre-processing step is essential to reduce the dynamic range in the contaminated image to a reasonable level for neural network training.
</ul> 

In the second step, we combine the input/output of two independently trained deep neural networks, Seg-UNet and Rec-UNet. We refer to this step as the SERENEt pipeline.

<ul>
  <li>  The former is a U-shaped 2D convolutional neural network for the identification of neutral hydrogen (HI) regions in 21-cm tomographic images. Results of SegU-Net can be found in our recent publication, <a href="https://academic.oup.com/mnras/article/505/3/3982/6286907?login=true">Bianco et al. (2021)</a>. 
  
  <li> By combining the residual and the prior images, RecU-Net will extrapolate meaningful information from both fields for an enhanced recovery of the 21-cm image. This is implemented by convolution blocks that takes the prior image as an additional input and intercepts the skip connection between the encoder and decoder layers. We show the architecture in Figure 2.
</ul> 


that from produes segmentation maps that identify HI region 
<img src="https://github.com/micbia/serenet/blob/main/docs/RecUNet_model.png"> <br/>
Figure 2: An overview of the architecture of RecU-Net. Convolutional layers process the binary map provided by SegU-Net, employed as a shape and position prior for region of 21-cm emission, and intercept the skip connection between the decoder and encoder.<br/><br/>


<b>Seg U-Net Training Utilization:</b></br>

<p style="margin-left:10px">&#9654; python segUnet.py config/net.ini</p>

<b>Seg U-Net Predicts 21cm:</b></br>

<p>The code is structured in folders that organize the different python scripts:</p>
<ul>
config/<br/>
├─ net_config.py<br/>
├─ net_RecUnet.ini<br/>
├─ net_Unet_lc.ini<br/>
├─ net_SERENEt_lc.ini<br/>
tests/<br/>
utils/<br/>
├─ 3Dto2D.py<br/>
├─ other_utils.py<br/>
utils_data/<br/>
utils_network/<br/>
├─ callbacks.py<br/>
├─ dataset.py<br/>
├─ data_generator.py<br/>
├─ metrics.py<br/>
├─ networks.py<br/>
utils_plot/<br/>
├─ other_utils.py<br/>
├─ plotting.py<br/>
├─ plot_optimisation.py<br/>
├─ plot_test.py<br/>
├─ postprops_plot.py<br/>
utils_pred/<br/>
├─ prediction.py<br/>
opt_talos.py<br/>
pred_serenet.py<br/>
serenet.py<br/>
predserenet.daint
</ul>


<p style="margin-left:40px">some text</p>
