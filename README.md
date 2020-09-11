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

* Bounding box is not rectangular. Eight instead of four corners
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
* Slower, needs more compute (masks have to be resized, and post prrocessed), also possibly more data transfer.

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
* `investigate_data.py` - does a rudimentary EDA (extracts some information from the imaes) and plots them in the downloaded `cropping_data` folder. It also draws the bounding boxes
* `prepare_data.py` - prepares the dataset by removing odd images, stratifying the dataset and creating the directory `data`, with subfolders `train`, `test`, `validation` and `demonstration`, as well as the csvs `train.csv`, `test.csv`, `validation.csv`. The csvs contain the bounding box information and the fullpath to the images. `Demonstration` contains the discarded images.
* `train_model.py` usess `train` and `validation` to train the neural network model, settings like learning rate and number of epochs are in the header.
* `evaluate_model.py` finally uses the model and predicts bounding boxes for the images in `test`, also drawing the ground truth and predictions on the images, saving them in `evaluate_images`, it also stores the `IOU` between the ground truth and the prediction in `score.csv`.

Then there is also `predict_cli.py` which can be used to predict images in a folder or given a filename. For example
```
predict_cli.py /data/demonstration/ model/mobilenetv2
```
Does image prediction in the demonstration folder.

**Warning in the current implementation, the predictions are written inside the input-path, and predictions are simply numbered (single file prediction might also be a problem).**

### Content of cropping_lib
cropping_lib.utils
