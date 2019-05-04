# Practice YOLO2 (Detection, Training, and Evaluation)

> This project is based on: https://github.com/experiencor/keras-yolo2

## Training guide:

#### 1. Data preparation 

Download the Raccoon dataset from from https://github.com/experiencor/raccoon_dataset.

#### 2. Edit the configuration file
You have to modify the parameters in the file `config.json`: ```labels```, ```train_image_folder```, ```train_annot_folder```, ```backend```.
To modify other parameters are optional.
The configuration file is a json file, which looks like this:
```json
{
    "model" : {
        "backend":              "Tiny Yolo",
        "input_size":           416,
        "anchors":              [4.03,5.23, 5.60,9.47, 7.53,11.54, 9.92,8.75, 11.02,11.99],
        "max_box_per_image":    10,        
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/minhnc-lab/WORKSPACES/AI/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/minhnc-lab/WORKSPACES/AI/data/raccoon_dataset/annotations/",
          
        "train_times":          10,
        "pretrained_weights":   "",
        "batch_size":           16,
        "learning_rate":        1e-4,
        "nb_epochs":            50,
        "warmup_epochs":        3,

        "object_scale":         5.0 ,
        "no_object_scale":      1.0,
        "coord_scale":          1.0,
        "class_scale":          1.0,

        "saved_weights_name":   "tiny_yolo_raccoon.h5",
        "debug":                true
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",

        "valid_times":          1
    }
}
```

**The list of supported backends, you can see in the file `frontend.py`. You have to download the backend that you defined in `config.json`, then copy it to the root folder of project. Otherwise, the code does not work.**

Download pretrained weights for backend at: https://1drv.ms/f/s!ApLdDEW3ut5fec2OzK4S4RpT-SU

#### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

#### 4. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

#### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.
