## MUStARD *Extended* Multimodal Sarcasm Detection Dataset

This repository contains the dataset and code for our ACL 2020 paper: 
[Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis](https://www.aclweb.org/anthology/2020.acl-main.401/)

### MUStARD Dataset
The original **MUStARD** dataset released in [Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)](https://www.aclweb.org/anthology/P19-1455/). The MUStARD dataset is a [multimodal video corpus](https://github.com/soujanyaporia/MUStARD) for research in automated sarcasm discovery. The dataset is compiled from popular TV shows including Friends, The Golden Girls, The Big Bang Theory, and Sarcasmaholics Anonymous. MUStARD consists of audiovisual utterances annotated with sarcasm labels. Each utterance is accompanied by its context, which provides additional information on the scenario where the utterance occurs.

### Extended MUStARD Dataset with Sentiment and Emotion Classes
We manually annotate this multi-modal **MUStARD** sarcasm dataset with *sentiment* and *emotion* classes, both implicit and explicit. You can download extended MUStARD datasets from [here](https://drive.google.com/drive/folders/1dJZyCSm80UZFHwbBRRg89njTDOwPkWa8?usp=sharing). 

### Model Description
* For multi-tasking, we propose two attention mechanisms, viz. Inter-segment Inter-modal Attention and Intra-segment Inter-modal Attention. 
* The main motivation of Inter-segment Inter-modal Attention is to learn the relationship between the different segments of the sentence across the modalities. 
* In contrast, Intra-segment Inter-modal Attention focuses within the same segment of the sentence across the modalities.

  To illustrate and document this format, we use a corresponding BibTeX entry taken and modified from a [real example](https://doi.org/10.5281/zenodo.15991)). Note that all data/software BibTeX entries should be of the `@misc` type: 

    @misc{lia_corrales_2015_15991,
        author       = {Lia Corrales},
        title        = {{dust: Calculate the intensity of dust scattering halos in the X-ray}},
        month        = mar,
        year         = 2015,
        doi          = {10.5281/zenodo.15991},
        version      = {1.0},
        publisher    = {Zenodo},
        url          = {https://doi.org/10.5281/zenodo.15991}
        }


    @inproceedings{mustard,
        title = "Towards Multimodal Sarcasm Detection (An  \_Obviously\_ Perfect Paper)",
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


