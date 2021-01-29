# palm_tree_detection

Palm tree detection using sliding window approach with HOG+SVM classifier.

Objective - Detect palm trees from an image

Approach - 
Slide rectangle of wxh dimension on image for candidate region for palm tree and then classify region as background/tree using SVM classifier which is trained with HOG feature of palm tree and background image.

How to run -
Training - Build for training

./tree_detection <path_of_negative_class_folder> <path_of_positive_class_folder> <path_of_model>


Inference - Build for inference

./tree_detection <path_of_model> <path_of_input_image> <path_of_output_image>


Training Detail ->
SVM classifier is trained for 64x64 size of images, ~800 positive and ~170 negative images are used.

Inference Detail ->
Input image is resized to 400x400.
Sliding window of wxh rect is used to generate crop region which is classified using svm model.
Three different scale is used for sliding window (54x54, 64x64, 74x74)
Output bounding box is grouped together for final predictions.
