# McKinsey Prohack Hackathon

## Brief
The aim of the [hackathon](https://prohack.org/) was to predict the composite index to measure the well-being of galaxies based on a number of features in the provided data and then allocate 50000 zillion DSML units of energy from a star across the galaxies based on certain contraints to optimise their well-being. 

## Part 1 - Regression problem

After looking through the dataset provided (EDA in notebook), we first had to find different way to deal with the null values and then tried multiple regression models to minimise the RSME. 

### Dealing with Null Values 

We first tried to fill the null values in through rules based on columns correlations ourselves. This included dropping columns with low correlation to y, using highest pair column correlations to fill in the nulls intelligently and so on (more detail in the notebook). 

We then decided to use the experimental scikit-learn Imputer class that uses a variety of methods to intelligently fill in the gaps. We did prepare the dataset such that we forward filled and back filled rows with 85% or more null values to ensure more accuracy before feeding it to the Imputer class. 

### Modelling 

We tried multiple regression models:

- Linear Reg 
- Lasso Reg 
- Ridge Reg 
- KNNRegressor: this was our best model with a train set MSE score of 0.029592
- Random Forest Regressor 
- CatBoost Regressor 
- Neural Networks 
- Support Vector Machine 
- XGBoost
- Stacking models
    - First layer: KNNRegressor, Lasso, XGBoost 
    - Second layer: XGBoost


## Part 2 - Optimisation task

According to the brief, in order to achieve the highest level of well-being through optimal allocation of the 50000 zillion DSML units from the star and satisfy the agreements between galaxies, we have to satisfy the following: 

- no one galaxy will consume more than 100 zillion DSML units and no galaxy should have below 0 zillion DSML units allocated
- at least 10% of the total energy will be consumed by galaxies in need with existence expectancy index below 0.7.

Every galaxy has a certain limited potential for improvement in the Index (our target variable) described by the following function:

- Potential for increase in the Index = -np.log(Index+0.01)+3

Likely index increase dependent on potential for improvement and on extra energy availability is described by the following function:

- Likely increase in the Index = extra energy * Potential for increase in the Index **2 / 1000

Our aim is to maximise the likely increase in the index. 

### Our solution

Given that the likely increase in Index for every galaxy is just a constant multiplied by the extra energy we are trying to allocate and that the potential for increase in the Index for each galaxy appears to be inversely proportionl to the expectancy index, we order the galaxies by the Index (our predicted variable) and allocate the first 500 galaxies, 100 zillion DSML units, with the remaining getting 0. 

All of the galaxies with an expectancy index below 0.7 are also in the 500 galaxies so this satisfies all of the constraints, while giving all the galaxies with the highest potential for increase in the Index the maximum they can recieve.

## Results and Scores 

The submission score is calculated on the test set using the following function:

80% prediction task RMSE + 20% optimization task RMSE * lambda where lambda is a normalizing factor

Our best score, based on the KNNRegressor and our optimiser, was 0.08222254.

