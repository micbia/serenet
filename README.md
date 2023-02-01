# SERENEt
SEgmentation and REcover NEtwork for SKA-Low multi-frequency tomographic 21-cm data for the Epoch of Reionization (EoR). The SERENEt code consist of a pre-process step, for foreground mitigation, and two U-shaped neural network for segmentation and 21-cm signal recovery, respectively. A general overview is given in Figure 1.<br/>
<img src="https://github.com/micbia/serenet/blob/main/docs/SERENEt_pipeline.png"><br/>
Figure 1: A simplified view of the project pipeline. The data include the mock observation with foreground and instrumental noise contamination, <i>I_obs</i>. The residual image after the pre-process step, <i>I_res</i>. The binary prior for neutral region identification and the recovered 21-cm image. Data input and output are shown with an example image.<br/><br/>

<ul>
  <li> In the first part, before applying any machine learning method, we process the challenge input data, <i>I_obs</i> , with an algorithm that partially subtracts the foreground contamination. The resulting residual image, <i>I_res</i> , will still contain some foreground residual and most systematic noise. However, this pre-processing step is essential to reduce the dynamic range in the contaminated image to a reasonable level for neural network training.
</ul> 

In the second step, we combine the input/output of two <b>independently trained U-shaped 2D convolutional neural network</b>, Seg-UNet and Rec-UNet. We refer to this step as the SERENEt pipeline.

<ul>
  <li>  The former is a segmentation for the identification of neutral hydrogen (HI) regions in 21-cm tomographic images. Results of SegU-Net can be found in our recent publication, <a href="https://academic.oup.com/mnras/article/505/3/3982/6286907?login=true">Bianco et al. (2021)</a>. The resulting binary image, <i>I_B</i>, is employed as a shape and position prior for region of 21-cm emission for the second and final component of SERENEt that aim to recover the 21-cm signal.
  <li> By combining the residual and the prior images, RecU-Net will extrapolate meaningful information from both fields for an enhanced recovery of the 21-cm image. This is implemented by convolution blocks that takes the prior image as an additional input and intercepts the skip connection between the encoder and decoder layers. We show the architecture in Figure 2.
</ul> 

<img src="https://github.com/micbia/serenet/blob/main/docs/RecUNet_model.png"> <br/>
Figure 2: An overview of the architecture of RecU-Net. Convolutional layers process the binary map, provided by SegU-Net, and intercept the skip connection between the decoder and encoder.<br/><br/>

### Configuration files:
The configuration file stores the initial condition for the network training. The file is located in <b>config/</b> with extension <i>.ini</i>.<br/>
Some of the variables are self-explanatory, while the following need to be changed accordingly:
<ul>
    <li><u><b>AUGMENT</b></u>: [string] the network architecture to use.
    <li><u><b>CHAN_SIZE</b></u>: [int] channel dimension of the low-latent dimensional space.
    <li><u><b>PATH_IO</b></u>: [string] the location of the training and validation sets.
    <li><u><b>SCRATCH_PATH</b></u>: [string] the location where to store the network training outputs.
    <li><u><b>DATASET_PATH</b></u>: [string or tuple string] the name of the training and validation sets.
    <li><u><b>LOSS</b></u>: [string] the name of loss function to be used (conform to the available metris in <i>metrics.py</i>).
    <li><u><b>METRICS</b></u>: [string or list string] the name of the training and validation sets.
    <li><u><b>GPUS</b></u>: [bool] if true it uses all the available GPU devices on the machine (deprecated, soon to be removed).
</ul> 
 
### Networks Training:
To train the network on your dataset, change the directory path variable <i>PATH_IO</i> in the initial condition files. The actual data should be stored at this location in <i>DATASET_PATH/</i>, in a sub-directory called <i>data/</i>.<br/>
To run use the following command:
<p style="margin-left:10px">&#9654; python serenet.py config/net.ini</p>
The code copy and updated the <i>.ini</i> file in the output directory. If you require to resume the training, you should change the second arguments to the corresponding file in the defined output location.<br/>
In <i>segunet.daint</i> you can find an example of a bash shell for submitting trianing jobs on the <a href="https://www.cscs.ch/computers/piz-daint/">Piz Daint</a> machine at CSCS.

### Network Predicting:
This section is still under development. You can have a look at the bash script <i>predserenet.daint</i> and the two python scripts <i>pred_recUNet.py</i> and <i>pred_segUNet.py</i> for reference.

### Code Structure:
<p>The SERENEt code is structured in folders that organize the different python scripts for training, prediction or post-processing plotting. Here a list of the most relevant files:</p>
<p style="margin-left:40px">
    config/<br/>
    ├─ net_config.py<br/>
    ├─ net_RecUnet.ini<br/>
    ├─ net_SegUnet_lc.ini<br/>
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
</p>


