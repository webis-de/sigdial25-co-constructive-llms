# Co-Constructive LLMs evaluation

## Create dataset from database

Run ``python create_dataset.py`` to create a dataset in json format from the user study sqlite3 database. The dataset is stored in ``user_study_data/``.
## Evaluate questionnaires and chats

Run ``python evaluate_chats.py`` to get the results of the questionnaires and the chat statistics stored in ``results/``. In addition, txt files of each chat are created and stored in ``chats/``. 

## Create histogram plots

Run ``create_plots.ipynb`` to create the histogram plots for the objective comprehension and enabledness questionnaire performances. The plots are stored in ``plots/``.

## Create tables

Run ``create_latex_tables.ipynb`` to create the histogram plots for the objective comprehension and enabledness questionnaire performances.

## Calculate correlations

Run ``calculate_correlation.ipynb`` to calculate possible correlations between the questionnaires results.


## Select cases for qualitative analysis

Run ``python select_chats_for_qualitative_analysis.py`` to filter for possible interesting chats for the qualitative analysis. The selected cases are stored as txt file in ``qualitative_analysis/``.