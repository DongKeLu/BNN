
## Introduction
Macroeconomic policy-making (by central banks and other government agencies) often relies on economic models, which are typically assumed to be linear or simple nonlinear forms. For example, the Phillips curve is often assumed to be linear, but its predictive power is weak, usually attributed to structural changes in the coefficients or other nonlinear forms. An increasing number of researchers are using nonlinear parametric models to estimate the Phillips curve. However, the main obstacle to using these flexible models in policy institutions is that they usually require strong assumptions about the nature of nonlinearity and a large amount of prior information. Moreover, determining whether a nonlinear form is appropriate is also a key issue.

Modern deep learning techniques can be applied to learn functional forms from data. However, the use of neural networks in policy institutions is limited. Neural networks are designed for datasets with millions of observations and covariates, while macroeconomic datasets have only a few hundred observations and almost as many time series. Therefore, specifying and estimating complete multilayer neural networks on these datasets is challenging, making their application in empirical analysis and policy-making complex. To address these challenges, this paper develops a Bayesian Neural Network (BNN) suitable for macroeconomic analysis.

## Advantages of BNN
- **Handling Small Sample Data**: Traditional neural networks require large datasets for training, while macroeconomic datasets often have limited observations. BNNs, by introducing the Bayesian framework, can effectively handle small sample datasets and avoid overfitting.
- **Model Flexibility**: BNNs possess strong nonlinear modeling capabilities, enabling them to capture complex nonlinear relationships. This makes them more advantageous than traditional linear models when dealing with nonlinear patterns in macroeconomic data.
- **Uncertainty Quantification**: Bayesian methods provide uncertainty estimates for parameters and predictions. This is crucial for policymakers who need to understand the reliability of the prediction results.
- **Avoiding Complex Model Selection**: By using shrinkage priors such as the horseshoe prior, BNNs can automatically prune irrelevant neurons, reducing the workload and subjectivity involved in manually selecting model structures.
- **Superior Prediction Performance**: In applications with both simulated and real data, BNNs demonstrate more accurate point and density prediction capabilities compared to other machine learning methods.

## Data
Four data files are used in this project:
- `current.csv`: Feature data
- `CE16OV.csv`: Employment data
- `CPIAUCSL.csv`: CPI data
- `INDPRO.csv`: Industrial production data

For numerical data, linear interpolation and mean imputation are combined; for non-numerical data, mode imputation is used. The target variables (employment, CPI, industrial production) are tested for stationarity. If a series is non-stationary, it is differenced, and the differenced data is saved to `differenced_target_variables.csv` for subsequent analysis.

The feature data undergoes Principal Component Analysis (PCA). The data is first standardized to eliminate scale effects, and then 8 principal components are extracted, with the results saved to `pca_components.csv`.

Both the feature data and differenced target variables are standardized. The standardized feature data is saved to `standardized_features.csv`, and the standardized target variables are saved to `standardized_targets.csv`. Additionally, the metadata of the target variables (including initial values and standardizers) is saved to `targets_metadata.pkl`, and the joint standardizer for the target variables is saved to `target_scaler.joblib` for use in model training and prediction.

## Code
The main code implements a complete BNN modeling and prediction workflow. It first loads the preprocessed macroeconomic data, including 8 feature variables and 3 target variables (employment, CPI, industrial production). These data have undergone rigorous preprocessing steps, including missing value handling, standardization, and differencing, to ensure the quality of model inputs. The feature data is reduced to 8 principal components through PCA, and the target variables are standardized and differenced to meet the requirements for model training.

The model adopts a three-layer neural network structure, comprising an input layer, two hidden layers, and an output layer. The first and second layers use the sigmoid activation function to introduce nonlinearity. The model is implemented using the Pyro framework and employs the No-U-Turn Sampler (NUTS) for Markov Chain Monte Carlo (MCMC) sampling to obtain the posterior distribution of model parameters. During training, the model parameters (including network weights and biases) and the standard deviation of the error term (sigma) are estimated simultaneously.

After training, the code performs detailed statistical analysis and visualization of the posterior distribution of model parameters, outputting the mean and standard deviation of each parameter and plotting parameter distribution histograms. These statistics and visualizations help us understand the uncertainty of model parameters and provide support for model interpretability.

In the prediction phase, the code uses the trained model to make predictions on all data, generating prediction samples and calculating the Root Mean Square Error (RMSE) on the standardized scale to evaluate the model's prediction performance. To transform the prediction results back to the original scale, the code implements inverse standardization, converting the standardized prediction values to their original scale values and calculating the RMSE on the original scale. The prediction results are presented in the form of mean, standard deviation, and 95% confidence intervals, providing more intuitive prediction information for decision-makers.


Additionally, the code predicts the values of the target variables for the next month and performs inverse transformation to restore them to the original scale. The prediction results include mean prediction, standard deviation, 95% confidence intervals, and percentage changes compared to the current and initial values. To enhance the visualization of the results, the code plots prediction interval figures on both the standardized and original scales, as well as the distribution of next month's predictions and a comparison of historical data with predictions. These visualizations provide an intuitive display of the model's prediction performance and uncertainty.

## Conclusion
The three-layer BNN model performs remarkably well in predicting macroeconomic variables. During model training, the posterior distribution of parameters is obtained through MCMC sampling and subjected to detailed statistical analysis and visualization. The prediction results show low RMSE on both the standardized and original scales, indicating high prediction accuracy. Moreover, the prediction intervals cover the actual values, demonstrating the model's advantage in uncertainty quantification. Future work can focus on further optimizing the model structure and hyperparameters to enhance prediction accuracy and exploring more applications of macroeconomic variables.
