# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Classification model, predicting the salary of the person based on the features like age, education, occupation, etc.  
The most current version of the model is saved under `.starter/model/model.pkl` file.  
The model is an gradient boosted classifier, developed using XGboost library. This is due to excellent handling of 
tabular data by that type of model.

## Intended Use

The model is intended to be used for predicting the salary of a person based on socioeconomical factors. Bear in mind
that the model is based on a singular point in time and doesn't adjust for time related factors like inflation.

## Training Data

80% of the census dataset. Some basic input preprocessing is done, like removing trailing whitespaces and converting 
categorical features to one-hot encoded vectors.

## Evaluation Data

Remaining 20% of the census dataset, not included in the training sample described above

## Metrics
Model metrics at the time of training are as follows:
 * **precision**=0.78
 * **recall**=0.66
 * **fbeta**=0.71

## Ethical Considerations

The model does take into account socioeconomic factors as input variables. Because of it, it might be
biased towards certain groups of people. For example, the model might be biased towards people of certain race
or country of origin, which reflects the trends in the data.  
Using the model might perpetuate said biases and must be approached with care.

## Caveats and Recommendations

The model is based on a singular point in time and doesn't adjust for time related factors like inflation.
