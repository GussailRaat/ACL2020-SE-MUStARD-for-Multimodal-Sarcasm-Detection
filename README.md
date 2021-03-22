## MUStARD *Extended* Multimodal Sarcasm Detection Dataset

This repository contains the dataset and code for our ACL 2020 paper: 
[Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis](https://www.aclweb.org/anthology/2020.acl-main.401/)

### MUStARD Dataset
The original **MUStARD** dataset released in [Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)](https://www.aclweb.org/anthology/P19-1455/). The MUStARD dataset is a [multimodal video corpus](https://github.com/soujanyaporia/MUStARD) for research in automated sarcasm discovery. The dataset is compiled from popular TV shows including Friends, The Golden Girls, The Big Bang Theory, and Sarcasmaholics Anonymous. MUStARD consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which provides additional information on the scenario where the utterance occurs.

### Extended MUStARD Dataset with Sentiment and Emotion Classes
We manually annotate this multi-modal **MUStARD** sarcasm dataset with *sentiment* and *emotion* classes, both implicit and explicit. You can download extended MUStARD datasets from [here](https://drive.google.com/drive/folders/1dJZyCSm80UZFHwbBRRg89njTDOwPkWa8?usp=sharing). 

### Data Format

    | Key  | Value |
    | ------------- | ------------- |
    | Content Cell  | Content Cell  |
    | Content Cell  | Content Cell  |


### Run the code
    python2 trimodal.py

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
