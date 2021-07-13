# Computer-Vision-practice

### Useful documentation:
- Loading and working with image datasets for classification: https://www.tensorflow.org/tutorials/load_data/images
- Data augmentation methods with `tf.keras.layers.experimenal.preprocessing`: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing

### General Notes:
- CNN Classifier model: **Images** (input) &rarr; **Base** (feature extraction) &rarr; **Head** (classification) &rarr; **Class** (output)

- The feature extraction performed by the base consists of three basic operations:

  1. *Filter* an image for a particular feature (convolution)
  2. *Detect* that feature within the filtered image (ReLU)
  3. *Condense* the image to enhance the features (maximum pooling)

- Make sure to use buffered prefetching to yield data from disk without having I/O become blocking. These are two important methods you should use when loading data:

  `.cache()` keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model.
  If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.

  `.prefetch()` overlaps data preprocessing and model execution while training.
  
  ```python
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
  ```python
  image_path = 'path/to/image.jpeg'
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
  
  
 - Capture and display webcam video with OpenCV code:
 
  ```python
  import cv2
  
  cap = cv2.VideoCapture(0)  # using webcam 0
  
  while True:
    success, frame = cap.Read()  # get video frame as image array
    
    # Put any img processing code here
    
    cv2.imshow("Video", frame) # Display frame
    cv2.waitKey(1)  # Displaying frame for 1ms
   ```
   
  - Resize output video window size code:
 
  ```python
  
  '''Capture video frame or img code'''
  
  cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)  # Create cv2 window
  cv2.resizeWindow('Resized Window', 1500, 1000)  # Resize created window
  
  cv2.imshow('Resized Window', name_of_retrieved_frame_var)  # Display your img in the created resized window
  ```
  OR change the size of video capture
  ```python
  cap_width, cap_height = 750, 700  # Define dimensions
  
  capture = cv2.VideoCapture(0)
  capture.set(3, cap_width)  # id 3 => capture window width
  capture.set(4, cap_height)  # id 4 => capture window height
  ```
