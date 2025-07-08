# Investigating Co-Constructive Behavior of Large Language Models in Explanation Dialogues

Code for the paper [Investigating Co-Constructive Behavior of Large Language Models in Explanation Dialogues](https://arxiv.org/abs/2504.18483).

For details on the approach, architecture and idea, please see the published paper.

```
@inproceedings{fichtel2025sigdial,
    title           = {Investigating co-constructive behavior of large language models in explanation dialogues},
    author          = {Leandra Fichtel and Maximilian Spliethöver and Eyke Hüllermeier and Patricia Jimenez and Nils Klowait and Stefan Kopp and Axel{-}Cyrille {Ngonga Ngomo} and Amelie Robrecht and Ingrid Scharlau and Lutz Terfloth and Anna-Lisa Vollmer and Henning Wachsmuth},
    booktitle       = {Proceedings of the 26th Annual Meeting of the Special Interest Group on Discourse and Dialogue},
    year            = 2025,
    publisher       = {Association for Computational Linguistics},
}
```

---

In each of the three sub-directories, you will find the code for the different components of the study. Please follow their respective README files for instructions on how to reproduce each component.

- [`./application`](./application): This directory contains the application that was build to conduct the study (i.e., the questionnaires and the interaction with the LLM).
- [`./evaluation`](./evaluation): This directory contains the code for the evaluation of the collected data.
- [`./turn-label-predictions`](./turn-label-predictions): This directory contains the code to train the turn label prediction model originally proposed by [Alshomary, et al. 2024](https://aclanthology.org/2024.lrec-main.1007/).



## Data

The data collected in our study and used for the evaluation can be found in a separate repository (due to a different license): https://github.com/webis-de/sigdial25-co-constructive-llms-data

- To use the data, simply add the `user_study_data` directory of the data repository into the `evaluation/` directory of this repository.
- Similarly, add the `final_selection_for_qualitative_analysis_25%` directory of the data repository into the `evaluation/qualitative_analysis/` directory of this repostory.
- Lastly, add the `final_mace_predictions_longformer-base-4096.pkl` file of the data repository into the `turn-label-predictions/data/` directory of this repository.



## Pre-trained models

The pre-trained models for the turn label prediction task can be found on huggingface: https://huggingface.co/webis/sigdial25-co-constructive-llms

To use the models, simply add the `final-turn-label-models` directory of the model repository into the `turn-label-prediction/data/` directory of this repository.
