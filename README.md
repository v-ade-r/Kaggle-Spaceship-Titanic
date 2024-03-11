# Kaggle-Spaceship-Titanic

Code for the "Spaceship Titanic" competition (classification problem) on Kaggle, guaranteeing 80%+ accuracy of predictions.

You can learn from this code how to:
1. **Manipulate a pandas DataFrame**; how to fill NaNs based on other columns and their properties or based on discoveries of specific patterns; how to iterate over DataFrame.
2. **Visualize the correlation** between particular values of a feature and the label. - Function **correlation_BarPlot** is a very neat and effective tool, showing not only bars, but absolute values, and percentages for every part of the bar.
3. **Create new feature**s.
4. **Tune hyperparameters** of the most efficient ML models for classification (with Optuna).

The accuracy is good but not excellent. To be honest I might have gone a little bit too far with the process of semi-manual filling NaNs, focusing sometimes on questionable patterns. It is obvious that people who chose easier methods like filling NaNs with SimpleInputer got better results. Nevertheless, for me, the goal of this competition was to have fun with the DataFrames and learn how to manipulate them effectively in practice, which I accomplished.
