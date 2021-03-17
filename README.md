## Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis
Code for the paper [Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis](https://www.aclweb.org/anthology/2020.acl-main.401/) (ACL 2020)

### Dataset:
We, at first, manually annotate the recently released multi-modal MUStARD sarcasm dataset with sentiment and emotion classes, both implicit and explicit. For multi-tasking, we propose two attention mechanisms, viz. Inter-segment Inter-modal Attention (Ie-Attention) and Intra-segment Inter-modal Attention (Ia-Attention). The main motivation of Ie-Attention is to learn the relationship between the different segments of the sentence across the modalities. In contrast, Ia-Attention focuses within the same segment of the sentence across the modalities. Finally, representations from both the attentions are concatenated and shared across the five classes (i.e., sarcasm, implicit sentiment, explicit sentiment, implicit emotion, explicit emotion) for multi-tasking.

For the evaluation of our proposed multi-task framerwork, we use benchmark multi-modal dataset i.e, MOSEI which has both sentiment and emotion classes.

### Dataset

* You can download datasets from [here](https://drive.google.com/open?id=1kq4_WqW0tDzBLu01yZbvdCpQ0iPBJWyQ).

* Download the dataset from given link and set the path in the code accordingly make two folder (i) results and (ii) weights.
