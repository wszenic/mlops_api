# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Classification model, predicting the salary of the person based on the features like age, education, occupation, etc.

## Intended Use

The model is intended to be used for predicting the salary of a person based on socioeconomical factors. Bear in mind
that the model is based on a singular point in time and doesn't adjust for time related factors like inflation.

## Training Data

80% of the census dataset. Some basic input preprocessing is done, like removing trailing whitespaces and converting 
categorical features to one-hot encoded vectors.

## Evaluation Data

Remaining 20% of the census dataset

## Metrics
_Please include the metrics used and your model's performance on those metrics._

## Ethical Considerations

The model does take into account socioeconomic factors as input variables. Because of it, it might be
biased towards certain groups of people. For example, the model might be biased towards people of certain race
or country of origin, which reflects the trends in the data.  
Using the model might perpetuate said biases and must be approached with care.

## Caveats and Recommendations

The model is based on a singular point in time and doesn't adjust for time related factors like inflation.
