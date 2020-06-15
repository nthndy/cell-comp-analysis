# Analysis for reverse engineering cell competition using multimodal microscopy and deep learning analysis

This repository features various analysis scripts from my PhD project studying cell competition.

## Cell Competition
Cell competition is broadly defined as a quality control mechanism that results in less-fit “loser” cells being eliminated from a biological tissue by wild-type "winner" cells. This mechanism impacts a wide variety of different physiological and pathological scenarios, from senescence to tumorigenesis to tissue development.

### Methodology
I use time-lapse microscopy to image the behaviours of cellular populations through multiple generations in a competition scenario. Fluorescent markers and quantitative phase interferometry yield a high level of detail into the internal cellular dynamics leading up a certain fate committment, be it mitosis, apoptosis or senescence. Coupling this data with deep-learning powered cellular segmentation and Bayesian multiple-object tracking allows for detailed picture of the chronology of an individual cells cycle: when certain events are initiated and in what circumstances. I aim to use this data to understand what factors play influential roles in triggering cellular decisions. For example, do the “winner” cells that I am studying commit to division in order to crowd out the “loser” cells, or do they simply fill the void left by a “loser” cell that has chosen to apoptose? Is the cellular decision making active or passive? 

#### Scripts

Code here includes:

* Napari scripts for creating neural network training data
* Segmentation analysis scripts
* Plotting scripts for extracting fluorescence and interferometric information
* Bayesian tracking scripts with personalised parameters for interferometric microscope
