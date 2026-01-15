<h2>TensorFlow-FlexUNet-Image-Segmentation-Nepal-LandSlide (2026/01/15)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Nepal-LandSlide</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and  an Upscaled 512x512 pixels <a href="https://drive.google.com/file/d/1rFFfTHUbh_D5BG8dv2r_eBbeRSNiirqx/view?usp=sharing"><b>Augmented-Nepal-LandSlide-ImageMaskDataset.zip</b> </a> , 
which was derived by us from <br><br>
<a href="https://zenodo.org/records/3675407">
Nepal landslide dataset for semantic segmentation</a>
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original <b>Nepal landslide dataset</b>, which contains 265  images and their corresponding masks  respectively,
we generated  the  Augmented dataset by using a Python script <a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a>.
<br><br> 
<hr>
<b>Actual Image Segmentation for Nepal-LandSlide  Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10042.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10042.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10042.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10013.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10242.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10242.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10242.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://zenodo.org/records/3675407">
Nepal landslide dataset for semantic segmentation</a>
<br><br>
Bragagnolo, Lucimara; Rezende, Lujan Rafael; da Silva, Roberto Valmir; Grzybowski, José Mario Vicensi
<br><br>
This database contains images used for the semantic segmentation of landslide scars from a fully convolutional neural network U-Net.
<br>
<b>1. Training dataset:</b> it contains 230 GeoTIFF 8 bits images and associated PNG masks (scars indicated in white and background in black color).
<br>
<b>2. Validation dataset: </b>it contains 35 GeoTIFF 8 bits images and associated PNG masks used for U-Net validation step.
<br>
<b>3. Test dataset: </b>it contains 10 GeoTIFF 8 bits images and associated PNG masks for testing.
<br><br>
<b>Citation</b><br>
Bragagnolo, L., Rezende, L. R., da Silva, R. V., & Grzybowski, J. M. V. (2020). <br>
Nepal landslide dataset for semantic segmentation (Version 1) [Data set]. Zenodo.
<a href=" https://doi.org/10.5281/zenodo.3675407"> https://doi.org/10.5281/zenodo.3675407</a>
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">Creative Commons Attribution 4.0 International</a>
<br><br>
<h3>
2 Nepal-LandSlide ImageMask Dataset
</h3>
 If you would like to train this Nepal-LandSlide Segmentation model by yourself,
please download the master  dataset from
<a href="https://drive.google.com/file/d/1rFFfTHUbh_D5BG8dv2r_eBbeRSNiirqx/view?usp=sharing"><b>Augmented-Nepal-LandSlide-ImageMaskDataset.zip</b> </a>
, expand the downloaded, and put it under <b>./dataset</b> folder to be:
<pre>
./dataset
└─Nepal-LandSlide
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
         ├─images
         └─masks
</pre>
<br>
As shown below, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<b>Nepal-LandSlide Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Nepal-LandSlide/Nepal-LandSlide_Statistics.png" width="512" height="auto"><br>
<br>
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Nepal-LandSlide TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Nepal-LandSlide/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Nepal-LandSlide and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 2
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00006
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Nepal-LandSlide 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Nepal-LandSlide 1+1
;                  LandSlide: 
rgb_map = {(0,0,0):0, (200, 128, 0):1,}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (34,35,36)</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (69,70,71)</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 71 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/train_console_output_at_epoch71.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Nepal-LandSlide/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Nepal-LandSlide/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Nepal-LandSlide</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Nepal-LandSlide.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/evaluate_console_output_at_epoch71.png" width="880" height="auto">
<br><br>Image-Segmentation-Nepal-LandSlide
<a href="./projects/TensorFlowFlexUNet/Nepal-LandSlide/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Nepal-LandSlide/test was very low, and dice_coef_multiclass very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0045
dice_coef_multiclass,0.9977
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Nepal-LandSlide</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Nepal-LandSlide.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Nepal-LandSlide Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10013.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10009.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10055.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10055.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10055.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/10182.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/barrdistorted_1001_0.3_0.3_10128.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/barrdistorted_1001_0.3_0.3_10128.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/barrdistorted_1001_0.3_0.3_10128.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/images/barrdistorted_1002_0.3_0.3_10018.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test/masks/barrdistorted_1002_0.3_0.3_10018.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Nepal-LandSlide/mini_test_output/barrdistorted_1002_0.3_0.3_10018.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Landslide4Sense
</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-Hokkaido-Iburi-Tobu-Landslide</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Hokkaido-Iburi-Tobu-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Hokkaido-Iburi-Tobu-Landslide
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Sichuan-Landslide
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Japan-Landslide
</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
