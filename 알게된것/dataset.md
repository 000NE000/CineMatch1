## train / dev / test dataset
The training set is for learning, the validation set is for tuning, and the test set is for final evaluation

### train dataset
This is the largest subset of the data, used to train the model. The model learns patterns and relationships from this data by updating its parameters during training.
Example: The training set helps the model identify patterns like “insult” vs. “non-insult” in a text classification task


### dev dataset (validation)
This dataset is used during training to tune hyperparameters and evaluate the model’s performance iteratively. It acts as a “mock exam” to assess progress and prevent overfitting on the training data.
- Example: The validation set helps determine optimal hyperparameters like learning rate or regularization strength
### test dataset
This is an entirely separate dataset used only after training is complete to evaluate the final performance of the model. It provides an unbiased estimate of how well the model generalizes to unseen data.
- Example: The test set simulates real-world scenarios by containing examples not seen during training or validation