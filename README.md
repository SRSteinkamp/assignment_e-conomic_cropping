# E-conomic assignment

## Task

Use machine learning to find the four corners of a receipt on an image.

## My Assumptions

* Should be run on mobile, i.e. light network / not too much processing
* We know that there is a receipt on the image (no additional classification)
* Preprocessing
     * On device - assuming there are more efficient implementations on IOS, Android
    * Could be minimal (resizing / standardization)
    * If network on server less transfer.
* Postprocessing
    * Similar assumptions to postprocessing

* **Product** is the neural network / algorithm predicting the corners (Tensorflow can export it through Tensorflow Lite etc.)

## Dataset intricracies

* Bounding box is not rectangular. Eight targets for prediction (not four as typically)
* Different image types
* Few images, some not ideal
    * Bounding box not large enough
    * Two receipts

## Brainstorming solutions

### Bounding box regression

Using a neural network for bounding box regression, due to limited data using a pretrained architecture (here MobileNetV2), with a different head. Solves the task at hand directly (i.e. finds four corners), is most efficient, and even if on server has the lowest data throughput.

#### Potential Issues
* Not rectangular bounding box
    * No off-the-shelf loss functions
    * Data-augmentation would need more time investment

### Image segmentation

Implementing a UNet (also based on MobileNetV2), to create a binary mask of the image. Extracts the receipt, probably with less noise in it than BBox regression. On the other hand, data augmentation and loss functions are there.

#### Potential Issues
* Needs postprocessing, especially if we want the corners
* Slower, needs more compute (masks have to be resized, and post processed), also possibly more data transfer.

### Computer Vision (rule based)

Using a set of algorithms like background segmentation, edge detections and other to extract the receipt and infer the four corners.

#### Potential Issues
* Rule based - might not generalize, needs really good data
* Possibly less efficient than DNN solutions.

**Decision:** Let's try out simple bounding box regression, which is not too fancy.

## Implementation

### Installation
I wrote a very small library for this task, which should run following the `requirements.txt` file. After cloning the repository `cd` into the folder and run `pip install -e .` to install the library.

To reproduce my results, there are five scripts (which **must** be run in the directory where the repository is located). Scripts are stored in the `scripts` folder:

* `download_data.py` - download and unzips the data-set from Github
* `investigate_data.py` - does a rudimentary EDA (extracts some information from the images) and plots them in the downloaded `cropping_data` folder. It also draws the bounding boxes
* `prepare_data.py` - prepares the dataset by removing odd images, stratifying the dataset and creating the directory `data`, with subfolders `train`, `test`, `validation` and `demonstration`, as well as the csvs `train.csv`, `test.csv`, `validation.csv`. The csvs contain the bounding box information and the fullpath to the images. `Demonstration` contains the discarded images.
* `train_model.py` usess `train` and `validation` to train the neural network model, settings like learning rate and number of epochs are in the header.
* `evaluate_model.py` finally uses the model and predicts bounding boxes for the images in `test`, also drawing the ground truth and predictions on the images, saving them in `evaluate_images`, it also stores the `IOU` between the ground truth and the prediction in `score.csv`.

Then there is also `predict_cli.py` which can be used to predict images in a folder or given a filename. For example
```
predict_cli.py /data/demonstration/ model/mobilenetv2
```
Does image prediction in the demonstration folder. The `model/mobilenetv2` and `model/customnetv1` are already present in the repository for immediate use. Where the latter provides better results.

**Warning in the current implementation, the predictions are written inside the input-path, and predictions are simply numbered (single file prediction might also be a problem).**

**Note:** On Windows it might be necessary to run the scripts and CLI by prepending `python scripts\` to the functions.

## Content of cropping_lib
* `cropping_libs.utils.image_utils`
    * `load_image` - To load images
    * `preprocess_image` - To preprocess images
    * `scale_coords_down` - To normalize coordinates
    * `scale_coords_up` - To rescale coordinates
    * `create_mask` - Creates a binary mask from polygons.

* `cropping_libs.utils.prepare_utils`
    * `get_data` - Downloads dataset from URL
    * `make_set` - Creates datasets (copying, setting up .csv files)
    * `get_bbox_names` - Prints out the order of the bounding boxes
    * `check_working_dir` - Enforces the use of scripts in repro dir

* `cropping_libs.model_parts.predict`
    * `make_prediction` - Preprocesses list of image in batch and performs prediction.
    * `predict_files` - Grabs file from folder, initiates Keras model and does prediction.

* `cropping_libs.model_parts.classes`
    * `IOU_LargeBox` - Calculate loss for largest bounding box for a set of coordinates (both true and prediction)
    * `IOU_TwoBox` - Calculates IOU loss for two bounding boxes (bottom left, top right and bottom right, top left)
    * `CenterLoss` - Calculates MSE for center coordinates of largest bounding box
    * `CombinedLoss` - Calculate loss based on IOUs, center and MSE
    * `DataGenerator` - DataGenerator used during training, loads and preprocesses data

* `cropping_libs.model_parts.model_architecture`
    * `build_model_mobilenet` - Build the based model based on MobileNetV2
    * `build_model` - build custom CNN
