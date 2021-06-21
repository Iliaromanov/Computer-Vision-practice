# Computer-Vision-practice

Useful documentation for loading and working with image datasets for classification: https://www.tensorflow.org/tutorials/load_data/images

### General Notes:
- Basic overview of CNN Classifier model: **Images** (input) &rarr; **Base** (feature extraction) &rarr; **Head** (classification) &rarr; **Class** (output)

- The feature extraction performed by the base consists of three basic operations:

  1. *Filter* an image for a particular feature (convolution)
  2. *Detect* that feature within the filtered image (ReLU)
  3. *Condense* the image to enhance the features (maximum pooling)

- Make sure to use buffered prefetching to yield data from disk without having I/O become blocking. These are two important methods you should use when loading data:

  `.cache()` keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model.
  If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

  `.prefetch()` overlaps data preprocessing and model execution while training.
  
  ```
  # Sample Data Pipeline
  
  AUTOTUNE = tf.data.AUTOTUNE
  ds_train = (
      ds_train_
      .map(data_preprocessing_func)
      .cache()
      .prefetch(buffer_size=AUTOTUNE)
  )
  ds_valid = (
      ds_valid_
      .map(data_preprocessing_func)
      .cache()
      .prefetch(buffer_size=AUTOTUNE)
  )
  ```

- Loading an individual image to np array:
  ```
  image_path = 'path/to/image.jpg'
  image = tf.io.read_file(image_path)
  image = tf.io.decode_jpeg(image, channels=1)
  image = tf.image.resize(image, size=[400, 400])
  img = tf.squeeze(image).numpy()
  ```
  
- When loading training dataset images with `tf.keras.preprocessing.image_dataset_from_directory`, the keras model adds batch_size to input shape. E.g: given input images of
  shape `(255, 255, 3)`, the model would use `input_shape=(batch_size, 255, 255, 3)`. 
  
  So when making predictions with `model.predict` **don't forget to add the batch_size value as shape[0] of input shape**. Eg: If you want to make prediction on a single image
  of shape `(255, 255, 3)`, use `model.predict({image-tensor}.numpy().reshape(1, 255, 255, 3)`
  
  If not done, then this this error is raised:
  
  `ValueError: Input 0 is incompatable with layer conv2d: expected ndim={n}, found ndim={n - 1}`
