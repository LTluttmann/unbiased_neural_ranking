Unbiased Neural Ranking

The main part of this repo is the mt.models module. It comprises three submodules, lse, ltr and ultr.

# LSE (Latent Semantic Entities)
This sub-module contains the encoder architectures and learning algorithms to map raw texts into a latent space

# LTR (Learning-to-Rank)
This sub-module contains the relevance scoring functions. Besides a simple MLP and a context-aware Attention-based ranker (attnrank), it also includes a simple "benchmark" model to imitate the production ranker, by taking the reciprocal of a documents rank position from the training / validation set as well as the interaction-based neural ranking models DSSM and KNRM. 

# ULTR (Unbiased LTR)
This sub-module contains learning strategies to isolate position bias present in clickthrough data. Besides the IPW framework of Joachims et al. (2017), for which propensity scores can be obtained by the EM algorithm implemented in this submodule, it contains the JoE architecture and a naive estimator, which simply uses the raw feedback signals from the training data.

Moreover, the module mt.data contains the scripts used for preprocessing and generating the dataset, except for the SQL used to get the clickstream data, as well as helper function to build the tensorflow datasets, which are used to train and evaluate the models. And the mt.evaluation model contains the scripts for a simulation study, that can be used to evaluate the effectiveness of the Unbiased LTR methods. 

In the notebooks folder, jupyter notebook scripts that were train and validate the models in this repo can be found. 

