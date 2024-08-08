
# Labeled Faces in the Wild Project (CNN)

This project involves the use of Convolutional Neural Networks (CNN) to recognize faces using the Labeled Faces in the Wild (LFW) dataset.

## Dataset Description

The LFW dataset is a collection of labeled face images designed for studying the problem of unconstrained face recognition. The dataset contains more than 13,000 images of faces collected from the web, with each face labeled with the name of the person pictured.

## Project Structure

The project includes the following files:

- **Labeled_Faces_in_the_Wild_CNN.ipynb**: Jupyter Notebook containing the data exploration, preprocessing, and model development using CNN.
- **requirements.txt**: List of dependencies required to run the project.

## Data Exploration

- **Imbalance**: The dataset is imbalanced with the majority of photos corresponding to George W. Bush, while Ariel Sharon is the least represented.
- **Normalization**: The data is already normalized, so further scaling by 255 is not required.

## Data Preparation

The following steps are involved in data preparation:

1. **Train-Test Split**: Separating the data into training and evaluation sets.
2. **Scaling**: Scaling the data using `StandardScaler()`.
3. **Encoding**: Converting the labels into categorical values using One-Hot-Encoding.

## Convolutional Neural Network (CNN) Model

The CNN model includes:

1. **Convolutional Layers**: Two convolutional layers with ReLU activation.
2. **Max Pooling**: Followed by a max pooling layer.
3. **Dropout**: A dropout layer to minimize overfitting.
4. **Flattening**: Flattening the feature maps before feeding into dense layers.
5. **Dense Layers**: Dense layers with dropout to improve generalization.

## Libraries Used

- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib

## Conclusion

The project demonstrates the use of CNN for face recognition using the LFW dataset. The model is capable of distinguishing between different individuals despite the imbalance in the dataset.

## Acknowledgements

- The dataset is publicly available on the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).

I hope you find this project interesting and useful. If you have any questions or suggestions, feel free to reach out.
