# medbioinfo_project

## Project

The project is the examining part of the course. Together with participation at the workshop it is compulsory to gain the course credits. The workload is expected to be about a week. The project is your chance to learn a bit more about some particular ML methods. The project consists of applying some ML metods to a particular dataset or datasets, and the compare the results. The results should be compiled in a written report including:

    Description of the chosen methods. In order to compare performance you need either to choose two (or more) different methods or in case of deep learning you could compare different architectures.
    What parameters are important to optimize for the chosen ML methods
    Which performance measures will be used, correlation, PPV, F1 or AUC? Does it matter?
    Description of your data set.
    Description of how cross-validation was performed. How was the data split to avoid similar examples in training and validation?
    Results from parameter optimizations, plots or tables.
        What parameters are optimal?
    Conclusions on the difference between ML methods, performance, sensitivity to parameter choices, ease-of-use etc.

If you cannot find a suitable ML project within your particular domain, you can use data from ProQDock, paper: https://academic.oup.com/bioinformatics/article/32/12/i262/2288786. Or you can choose a data set from the Machine Learning Repository To study. Make sure it has a good balance between number of examples (# Instances) and number of features (# Attributes).

For example:

    Epileptic Seizure Recognition Data Set
    Mice Protein Expression Data Set

Upload a pdf of your report as answer to the Project assignment in canvas

## Conclusions

After running ```jupyter notebook ML\ Medbioinfo\ Project.ipynb```
and going through the project outline the drawn conclusions were the following:
- The set is heavily biased on incorrect protein models, with many rows with zero in the DockQ-binary column. This certainly influence the result if we choose total accuracy since predicting all zeros will give a too high accuracy. It would be maybe better then to focus on precision and recall which only considers positive predictions.
- For the hyperparameters, we want to have the optimium value inside the searched grid, and not at the border. If the optimum is at the border, it would be better to go higher or lower to establish a maximum. Here the metric used for this analysis was accuracy.
