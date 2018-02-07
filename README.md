# Fusion Naive Bayes Classifier
## Implementation of fusion Naive Bayes classifier from scratch.

### The model
The model is based on classical Naive Bayes classifier. Classes probabilities is calculated using prior, likelihood and evidence.

### The implementation
Implementation of fusion Naive Bayes is done using only low-level computation library such as pandas, numpy and math. 
This implementation is an alternate to scikit-learn implementation of Naive Bayes classifier.

### The advantages
There are some advantages of the Fusion Naive Bayes classifier, over the scikit-learn Naive Bayes implementation

- Inputs to the model are pandas DataFrame, which is normally the start point for any data analyse pipeline
- Categorical variables/columns are processed natively. There is no need for categorical encoding. Scikit-learn version of Naive Bayes classifier requires input training data to be in the form of numpy.ndarray, which in turns requires all categorical variables/column must be encoded to numerical values. Encoding techniques can be one-hot, feature hashing, binary, ordinal... Hence, all encoding techniques may introduce noise/irrelevant information to the data, and cannot preserve the original data distribution. This is the main advantages of Fusion Naive Bayes over scikit-learn Naive Bayes
- Support for online/increment learning for all gaussian distribution, multinomial distribution and bernoulli distribution.
- Automatically selection of Gaussian, Multinomial and Bernoulli distribution, depend on variable types

### The usage:
Usage of Fusion Naive Bayes is similar to scikit-learn Naive Bayes, except for the input data format: dataframe, instead of numpy array

- Fit model:
```
model = Fusion_NaiveBayes()
model.fit(df_train_features, df_train_labels)
```

- Predict result: labels
```
prediction = model.predict(df_test_features)
```

- Predict result: probabilities
```
probs =  model.predict_proba(df_test_features)
```

- Predict result: log-probabilities
```
log_probs =  model.predict_log_proba(df_test_features)
```

- Performance assessment: results return by Fusion Naive Bayes are all numpy array, in similar dimension to result return by scikit-learn Naive Bayes. Therefore, all scikit-learn metrics can be used normmaly on Fusion Naive Bayes result.
```
from sklearn.metrics import accuracy_score
import numpy as np
accuracy = accuracy_score(np.array(df_test_labels), prediction)
```

- Online/Incremental learning:
```
model = Fusion_NaiveBayes()
model.fit(df_train_features, df_train_labels)
model.partial_fit(new_df_train_features,new_df_train_labels)
```

### The performance:
- To test model on iris dataset:
```
python test_fusion_naive_bayes iris
```
- To test model on census-income dataset:
```
python test_fusion_naive_bayes census-income
```

- Performance comparison: Iris dataset
> Fusion Naive Bayes: 0.96
> GaussianNB: 0.93
> MultinomialNB: 0.87

- Performance comparison: Census-income dataset
> Fusion Naive Bayes: 0.82
> GaussianNB: 0.77
> MultinomialNB: 0.77