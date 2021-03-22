### Raw Text

You can download datasets from [here](https://drive.google.com/drive/folders/1dJZyCSm80UZFHwbBRRg89njTDOwPkWa8?usp=sharing).

This link will only give the text file with extented annotation i.e., sentiment and emotion for both implicit and explicit.


For acoustic and visual, please click [here](https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view).


### Raw Videos
We provide a [Google Drive folder with the raw video clips](https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view), including both the utterances and their respective context

### Data Format
The annotations and transcripts of the audiovisual clips are available at [data/sarcasm_data.json](https://github.com/DushyantChauhan/ACL-2020-MUStARD-Extension/blob/main/data/sarcasm_data.json). Each instance in the JSON file is allotted one identifier (e.g. "1_60") which is a dictionary of the following items:

Example format in JSON:

{
  "1_60": {
    "utterance": "It's just a privilege to watch your mind at work.",
    "speaker": "SHELDON",
    "context": [
      "I never would have identified the fingerprints of string theory in the aftermath of the Big Bang.",
      "My apologies. What's your plan?"
    ],
    "context_speakers": [
      "LEONARD",
      "SHELDON"
    ],
    "sarcasm": true
  }
}
