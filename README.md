## MUStARD *Extended* Multimodal Sarcasm Detection Dataset

This repository contains the dataset and code for our ACL 2020 paper: 
[Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis](https://www.aclweb.org/anthology/2020.acl-main.401/)

### MUStARD Dataset
The original **MUStARD** dataset released in [Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)](https://www.aclweb.org/anthology/P19-1455/). The MUStARD dataset is a [multimodal video corpus](https://github.com/soujanyaporia/MUStARD) for research in automated sarcasm discovery. The dataset is compiled from popular TV shows including Friends, The Golden Girls, The Big Bang Theory, and Sarcasmaholics Anonymous. MUStARD consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which provides additional information on the scenario where the utterance occurs.

### Extended MUStARD Dataset with Sentiment and Emotion Classes
We manually annotate this multi-modal **MUStARD** sarcasm dataset with *sentiment* and *emotion* classes, both implicit and explicit. You can download extended MUStARD datasets from [here](https://docs.google.com/spreadsheets/d/1-qFklf5ZQPirUuQLF6GiyxJYB9YmENLrKrs-P-_r3HI/edit?usp=sharing) (text only). For rest of the modalities i.e., visual and acoustic, please follow this [GitHub repository](https://github.com/soujanyaporia/MUStARD).

### Data Format

| Key  | Value |
| ------------- | ------------- |
| utterance  | The text of the target utterance to classify. |
| speaker  | Speaker of the target utterance.  |
| context  | List of utterances (in chronological order) preceding the target utterance. |
| context_speakers  | Respective speakers of the context utterances. |
| sarcasm  | Binary label for sarcasm tag.  |
| implicit-sentiment  | Three labels for implcit sentiment tag.  |
| explicit-sentiment  | Three labels for explcit sentiment tag.  |
| implicit-emotion  | Nine labels for implicit-emotion tag.  |
| explicit-emotion  | Nine labels for explicit-emotion tag.  |


### Feature Extraction

There are two setups which are as follows;

#### (1) Speaker Dependent Setup (exMode=True)

* [datasetTrue_fasttext.zip](https://drive.google.com/file/d/1VYpcu4pkg30GUIjJ9P1L4gk9MnaKNm3b/view?usp=sharing): This file contains only text features (using fasttext 300d).

**Note**: see function *featuresExtraction_fastext(foldNum, exMode)* in *trimodal_true.py*, where foldNum belongs to [0-4] and exMode  = True


* [datasetTrue_original.zip](https://drive.google.com/file/d/1KsP__c28hQyBSanKDYNL-XwBfe18KJPX/view?usp=sharing): This file contains acoustic and visual features (from [here](https://github.com/soujanyaporia/MUStARD)).

**Note**: see function *featuresExtraction_original(foldNum, exMode)* in *trimodal_true.py*, where foldNum belongs to [0-4] and exMode  = True

--------------------
#### (2) Speaker Independent Setup (exMode=False)

* [datasetFalse_fasttext.zip](https://drive.google.com/file/d/1o9WvwSvpKbz_jbZuajGtlEQOrzUhIgR-/view?usp=sharing): This file contains only text features (using fasttext 300d).

**Note**: see function *featuresExtraction_fastext(foldNum, exMode)* in *trimodal_false.py*, where foldNum = 3 and exMode  = False


* [datasetFalse_original.zip](https://drive.google.com/file/d/1LyTp-3NsSPLbFt72ojpL0j2Reu0NA8VF/view?usp=sharing): This file contains acoustic and visual features (from [here](https://github.com/soujanyaporia/MUStARD)).
     
**Note**: see function *featuresExtraction_original(foldNum, exMode)* in *trimodal_false.py*, where foldNum = 3 and exMode  = False
     
     Download all the features and put into the folder **feature_extraction** and then run the code.

### Run the code
    
    python2 trimodal_true.py (for speaker dependent)
    
    python2 trimodal_false.py  (for speaker independent)

### Citation
Please cite the following paper if you find this dataset useful in your research:

    @inproceedings{mustard,
        title = "Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)",
        author = "Castro, Santiago  and
          Hazarika, Devamanyu  and
          P{\'e}rez-Rosas, Ver{\'o}nica  and
          Zimmermann, Roger  and
          Mihalcea, Rada  and
          Poria, Soujanya",
        booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = "7",
        year = "2019",
        address = "Florence, Italy",
        publisher = "Association for Computational Linguistics",
    }
    
    
    @inproceedings{chauhan-etal-2020-sentiment,
        title = "Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis",
        author = "Chauhan, Dushyant Singh  and
          S R, Dhanush  and
          Ekbal, Asif  and
          Bhattacharyya, Pushpak",
        booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
        month = jul,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.acl-main.401",
        pages = "4351--4360",
    }      
