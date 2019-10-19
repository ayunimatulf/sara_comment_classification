# sara_comment_classification
Developing a Logistic Regression model which can detect SARA comments automatically
### DATASET
Dataset include 2 categorical text, there are normal and SARA comments. Number of comments are  67221 and 17724, respectively. Percentage of data is represent in the diagram below. <br>
!['Data Precentage](https://github.com/ayunimatulf/sara_comment_classification/blob/master/data_diagram.png) <br>
From the diagram we know that there is a problem about the dataset because of unbalance the number of each category. So in this task I try to doing 2 method to doing 2 type classification task, the normal step in type-1 method and trying to handling unbalance data in type-2 method.
### DISCUSSION
In this discussion that will be split as 3 process, pre-processing data, prediction method, and implementation model.
1. Pre-processing data
    1. Import dataset and rewrite as pandas DataFrame with the label where 0 is normal (not SARA) and 1 is SARA
    2. (2 type) In this step I split the dataset to be 2 data, in data1 I use all of the dataset as the original dataset and in data2 I cut the normal dataset so the length between normal comments and SARA comments are same. I choose the comments randomly
    3. Then clean the data from non-letters characters, setting lower case in each sentence, and removing stopwords
    4. Create wordclouds from each categorical data to know what the frequent words appear in each category
    5. After that create doc-by-term matrix to represent each sentence as numerical vector, if A_(m×n) is doc-by-term matrix so each element in a_ij represent how many term_j appear in document_i
    6. Weighting each elements in matrix a using TF-IDF, the purpose of this step is to reflect how important a term is to a document in a collection or corpus.
2. Prediction method
Create the logistic regression model linier_model.LogisticRegression from sklearn. Set the multi_clas=’ovr’, it’s mean that the class for our data is binary class where the probabilities function that used for the classify is Sigmoid Function. The parameter is algorithm that used in the optimization problem. I used ‘sag’ or Stochastic Average Gradient descent solver as faster for large ones based on sklearn documentation. I also trying ‘saga’ and ‘newton-cg’ but there aren’t significant change in my evaluation metrics. First I training my model using training data and evaluate the data from testing data or validation data. 
3. Model Implementation
After I think I have the best model that I can raise I implemented the model for text input to knowing the text I have input is SARA or a normal one. The step is explained bellow :
    1. First I save my TF-IDF values from my dataset before so I can transform my new text to new doc_by_term matrix using old TF-IDF values
    2. Input a text
    3. Clean the text from non-letter characters, setting lowercase, and remove stopwords
    4. Transform the text to doc_by_term matrix using old TF-IDF value.
    5. Then I predicting the matrix using my model I have created
### RESULT
The evaluation metrics from data1 and data2 is represents respectively below :
<p align='center'>
  <img src='https://github.com/ayunimatulf/sara_comment_classification/blob/master/metric_report_data1.png'>
</p>
<p align='center'>
  <img src='https://github.com/ayunimatulf/sara_comment_classification/blob/master/metrics_report_data2.png'>
</p> <br>

From the evaluation metrics we know that Data1 has higher average precision, recall, and f1-score than data2 but the recall for label 1 is so low in data1. Low recall with high precision mean that the model can’t detect the SARA class well but is highly trustable when it does. In the data2 recall and f1-score for label 1 is better but for label is getting lower than before. Let’s see how if I input the text manually. <br>
<p align='center'>
  <img src='https://github.com/ayunimatulf/sara_comment_classification/blob/master/output.PNG'>
</p> <br>

For comment1 and comment2 the model has same output but for comment3 data1 cannot predict that the comment is SARA. 

