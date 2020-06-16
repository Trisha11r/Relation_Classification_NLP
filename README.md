# Relation_Classification_NLP
Given 10 predefined relations like cause-effect, product-producer, etc, the goal was to define the relation and the direction of the relation b/w 2 entities in a sentence.

For the advanced model, the CNN model has been implemented with one convolutional layer. The architecture details are described in Fig.3. The CNN model has been used as even after adding POS and dependency features the CNN model outperforms the biGRU model in the basic model. The paper mention in references (1), was referred for the implementation.
We define the convolution layer using keras layer implementation of Conv1D and max-pool using keras layer implementation of GlobalMaxPool1D. Define decoder using its keras layer implementation. In the call function, first the window processing is done using embedding_lookup for word and pos embeddings. Then we concatenate word and pos embeddings and then apply convolution on the concatenated output. Then max-pooling is done by calling the previously defined max_pool layer. At last, we extract sentence level features using the previously defined decoder.

READ ME for advanced model implemented
----------------------------------------------------------------------------------------------------------------------------------------
Model implemented: CNN (Convolutional Neural Network) with 1 convolution layer
----------------------------------------------------------------------------------------------------------------------------------------
Define the convolution layer using keras layer implementation of Conv1D and max-pool using keras layer implementation of GlobalMaxPool1D. Define decoder using its keras layer implementation.

In the call function, first the window processing is done using embedding_lookup for word and pos embeddings.
Then we concatenate word and pos embeddings and then apply convolution on the concatenated output.
Then max-pooling is done by calling the previously defined max_pool layer.
At last, we extract sentence level features using the previously defined decoder.
----------------------------------------------------------------------------------------------------------------------------------------
 
 From Fig.4, we can see from the F1 score values that the CNN model with 1 convolutional layer performs better than the Basic biGRU model implemented for word, POS and dependency features combined.
Three experiments conducted on this model are as follows:
1. Changing the regularization lambda to (10)^(-4):
a. Val loss for last epoch (09): 2.0366
b. Val F1 score: 0.5981
2. Training the model for word embeddings and dependency features:
a. Val loss for epoch: 2.0204
b. Val F1 score: 0.6044
3. Training the model for word embeddings and POS features:
a. Val loss for epoch: 2.323
b. Val F1 score: 0.5065
The prediction files for each have been uploaded.
