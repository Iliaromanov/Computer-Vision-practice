# Computer-Vision-practice

Useful documentation for loading and working with image datasets for classification: https://www.tensorflow.org/tutorials/load_data/images

### General Notes:
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
