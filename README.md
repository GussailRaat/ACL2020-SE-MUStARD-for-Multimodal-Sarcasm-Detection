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


<!--  https://scholar.google.com/intl/en/scholar/inclusion.html#indexing -->
<meta name="citation_title" content="Distill for R Markdown"/>
<meta name="citation_fulltext_html_url" content="https://rstudio.github.io/distill"/>
<meta name="citation_fulltext_world_readable" content=""/>
<meta name="citation_online_date" content="2018/05/04"/>
<meta name="citation_publication_date" content="2018/05/04"/>
<meta name="citation_author" content="JJ Allaire"/>
<meta name="citation_author_institution" content="RStudio"/>
<meta name="citation_author" content="Rich Iannone"/>
<meta name="citation_author_institution" content="RStudio"/>
<meta name="citation_author" content="Yihui Xie"/>
<meta name="citation_author_institution" content="RStudio"/>
<meta name="citation_reference" content="citation_title=Distill;
citation_publication_date=2016;citation_publisher=Distill Working
Group;citation_doi=10.23915/distill;citation_author=Shan Carter;
citation_author=Chirs Olah;citation_author=Arvind Satyanarayan"/>
<meta name="citation_reference" content="citation_title=Literate
programming;citation_publication_date=1984;
citation_publisher=British Computer Society;citation_volume=27;
citation_author=Donald E. Knuth"/>
<meta name="citation_reference" content="citation_title=Dynamic
documents with r and knitr;citation_publication_date=2015;
citation_publisher=Chapman; Hall/CRC;citation_author=Yihui Xie"/>
