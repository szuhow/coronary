# coronary

**models/**: Directory for trained models. 

**predict/**: Directory for prediction-related files - frames for checking models.

**seg_train/**: Directory containing training data, with additional subdirectories:

- **annotations/**: Directory for annotations.

- **images/**: Directory for training images.

- **masks/**: Directory for masks, used for training.

[Google drive - files](https://drive.google.com/file/d/1I9Nwq2kqqhiBYzR7a7eVd2BjDg7vZ13f/view?usp=drive_link)


To use the inference.py script, follow these examples:

Prediction Mode: To run the script in prediction mode, where the model generates predictions based on the provided data, use the following command:

```
python inference.py --epochs 50 --steps 2000 --mode predict
```

In this case, the script will use model file with 50 epochs and 2000 steps per epoch, operating in prediction mode (in this case: unet_model_50_epochs_2000_steps.h5).

Training Mode: To run the script in training mode, where the model is trained based on the provided training data, use the command below:

```
python inference.py --epochs 50 --steps 2000 --mode train
```

Here, the script will run with 50 epochs and 2000 steps per epoch, working in training mode.

In both cases, the `--epochs` parameter specifies the number of training or prediction epochs, `--steps` specifies the number of steps (batches of data) processed in each epoch, and `--mode` determines the script's mode of operation (`predict` for generating predictions, `train` for training the model).