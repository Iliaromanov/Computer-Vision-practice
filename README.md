# Computer-Vision-practice

Useful documentation for loading and working with image datasets for classification: https://www.tensorflow.org/tutorials/load_data/images

### General Notes:
- Basic overview of CNN model: **Images** (input) &rarr; **Base** (feature extraction) &rarr; **Head** (classification) &rarr; **Class** (output)

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
