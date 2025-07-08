# Co-constructive LLMs turn label predicition

## Train and evaluate turn label prediction model (k-fold cross validation)

Run ``./experiments/src-py/start_kfold_training.sh`` to train and evaluate a model to predict the dialogue act, the explanation move and the topic of a given turn text. The model of each fold is stored in ``data/turn-label-models/fold-<k>/<model_name>/``.

## Train final turn label prediction model

Run ``./experiments/src-py/start_final_training.sh`` to train a final model (whole dataset for training) to predict the dialogue act, the explanation move and the topic of a given turn text. The final model is stored in ``data/final-turn-label-models/<model_name>/``.

## Predict turn labels

Run ``experiments/src-ipynb/predicting-turn-labels-with-bert.ipynb``to predict the dialogue acts, explanation moves and topics of the turns of our user study using the final model. The predictions are stored in ``data/final_mace_predictions_<model_name>.pkl``.

## Evaluate predicted turn labels

Run ``data/turns_evaluation.ipynb`` to evaluate the distribution and possible correlations of the predicted dialogue acts and explanation moves. In addition, the text complexty of specific turn texts are calculated. The plots are stored in ``data/``. 

## Credits

The code, as well as the eli5 and five-levels datasets, are based on: https://github.com/MiladAlshomary/explanation-quality-assessment
