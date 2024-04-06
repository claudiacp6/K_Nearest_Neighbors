#  K-Nearest Neighbors (kNN)

The KNN, or "k-Nearest Neighbors," is an algorithm used for both classification and regression. It is known as an instance-based learning method or lazy learning method because it does not attempt to build an explicit model during the training phase. Instead, it memorizes the training examples and makes predictions based on the proximity (similarity) of the new examples to the existing examples.

The basic idea of k-NN is that examples in a similar feature space tend to have similar labels. That is, when predicting the label of a new example, the algorithm looks at the k nearest training examples to it in the feature space and assigns the most common class among these k neighbors to the new example.

This work develops a Python code "from scratch" and is divided into 4 parts.

## *1. Initialization Code in the KNN class*

       def __init__(self, k=3, normalize=False):
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
   
The init function defines 3 parameters:

 -**k**: Number of nearest neighbors to be considered by the k-NN algorithm (default is 3).
 -**Normalize**: A bool indicating whether the data should be normalized or not (default is False).
 -**Attributes**: The init method initializes some instance attributes with None values.


## *2. Fit Function*
 It is a method commonly used in machine learning algorithms to train the model based on the provided training data. In the context of k-NN (k-Nearest Neighbors), the fit method is responsible for storing the training data in the object instance so that it can be used later for making predictions.
 
       def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
   
  Where:
   - **x_train**: Is the training data set, containing the features of the training examples. Each row represents an example, and each column represents a feature.
   - **y_train**: Are the labels corresponding to the training examples in X_train. Each label is associated with a specific example and indicates the class or value that the model should learn.

Inside the fit method, the training data (X_train and y_train) is simply stored in the instance attributes self.X_train and self.y_train. This means that after calling the fit method, the class instance will have access to the training data, and the model can be trained based on this data.

## *3. Predict Method*
It is responsible for making predictions based on the provided test data.
    - X_test: Are the test data on which predictions will be made.

          def predict(self, X_test):
            if self.normalize:
             X_test = self.normalize_data(X_test)
          predictions = []
          for x in X_test:
             distances = np.linalg.norm(self.X_train - x, axis=1)
             indices = np.argsort(distances)[:self.k]
             k_nearest_labels = self.y_train[indices]
             unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
             predicted_label = unique_labels[np.argmax(counts)]
             predictions.append(predicted_label)
          return np.array(predictions)
        
The code used contains:

  -**Normalization** (if necessary): If the normalize attribute is True, the test data (X_test) is normalized using the normalize_data function before making predictions. This ensures that the test data is processed in the same way as the training data.

  -**Distance Calculation**: For each example in X_test, the distance is calculated with respect to all training examples in X_train. The np.linalg.norm function is used to calculate the Euclidean distance.

  -**Identification of Nearest Neighbors**: The indices of the k nearest neighbors are obtained using argsort to sort the distances and select the first k indices.

  -**Counting the Labels**: The occurrence of each label in the k nearest neighbors is counted.

  -**Assignment of Predicted Label**: The predicted label is determined by the most common label among the k nearest neighbors.

  -**Storage of Predictions**: The predictions are stored in the predictions list.

  -**Return of Predictions**: The predictions are returned as a NumPy array.

## *4. Data Normalization*
  It is used to normalize the data if the normalization option (self.normalize == True) is activated. Normalization is a common process in machine learning algorithms, where feature values are adjusted to ensure they are on the same scale.

       def normalize_data(self, data):
         min_vals = np.min(data, axis=0)
         max_vals = np.max(data, axis=0)
         return (data - min_vals) / (max_vals - min_vals)

Data normalization includes:

  -**Calculation of Minimums and Maximums**: For each column (feature) in data, np.min and np.max are used to calculate the minimum and maximum values.

  -**Normalization**: Each value in data is normalized by subtracting the minimum value from the corresponding column and dividing by the range (difference between the maximum and minimum).

  -**Return of Normalized Data**: The normalized data is returned by the function.

This method is used in the predict method before making predictions, if the normalization option (self.normalize) is activated.

# Example of Practical Application

In this example, example data is created for a binary classification problem.
Random numbers are generated by the np.random-seed(42) function. Later, a matrix of numbers between 0 and 1 is generated, with dimensions 100x2, where each row represents an example and each column represents a feature.




       np.random.seed(42)
       X_train = np.random.rand(100, 2)
       y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

       X_test = np.random.rand(20, 2)
       y_test = (X_test[:, 0] + X_test[:, 1] > 1).astype(int)


*y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)*: Calculates the sum of the elements in each row of X_train and checks if this sum is greater than 1. The result is a boolean array, which is then converted to integers (0 or 1) using astype(int). This creates the binary class labels based on the mentioned condition.

The model with the created data is trained with the fit function and the predictions are generated and stored in the variable predictions, with K=3.

       knn = KNN(k=3)
       knn.fit(X_train, y_train)

       predictions = knn.predict(X_test)

For graphical visualization:


       plt.figure(figsize=(10, 6))

       plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolor='k', marker='o', label='Treinamento')

       correct_predictions = predictions == y_test
       incorrect_predictions = ~correct_predictions

       plt.scatter(X_test[correct_predictions, 0], X_test[correct_predictions, 1], c='green', marker='^', s=100, label='Previsão Correta')
       plt.scatter(X_test[incorrect_predictions, 0], X_test[incorrect_predictions, 1], c='red', marker='v', s=100, label='Previsão Incorreta')


In summary, this code creates a visualization that shows the training points colored according to the classes, and the test points are marked differently depending on whether the model's predictions were correct or incorrect. The legend helps distinguish between correct and incorrect prediction points. This is useful for visually evaluating the performance of the k-NN model on the test data.

![image](https://github.com/claudiacp6/K_Nearest_Neighbors/assets/147619731/4a7090dd-e8f7-4fad-8309-29c3acf27415)




