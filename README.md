# code-completion

## Dataset creation

For the dataset creation part, I used six Python files from my projects, generating five samples
per file, resulting in total of 30 code completion examples. For each file, I selected random points
to represent a user’s cursor position and split the code into three parts to simulate a code
completion scenario. Files used:

1. _utils_models:_ ML models’ utils for preprocessing, visualization, and evaluation.
2. _quizbot._ py: simple python desktop app
3. _simple_cv._ py: simple computer vision task
4. _openai_gym_env._ py: implementation of OpenAI’s gym for a card game
5. _tinybert._ py: training of a TinyBert model
6. _aibg_hackaton._ py: heuristic driven bot

Each sample is stored as a JSON object with the following fields:

- _id:_ unique identifier for each sample
- _prefix:_ code appearing before the simulated cursor position
- _middle:_ the code assumed to be missing that model is expected to predict
- _suffix:_ the code following the cursor position (starting from line below)
- file: the name of the file from which the sample was extracted

Python script used for this: `dataset_creation.py`

## Generating Code Completions

The scripts loads an open-source code completion model [Tiny Starcoder](https://huggingface.co/bigcode/tiny_starcoder_py) to generate predictions
for code completions based on each dataset sample. For each sample, the model processes an
input prompt constructed from the prefix and suffix, leaving the middle section blank as the
target completion. The model then generates a prediction, limited to a maximum of 20 tokens.
The generated response is then processed to extract only the relevant middle part of the
completion by locating and trimming out everything but the middle part.


## Evaluation

In the evaluation process, I used several metrics to assess the quality of the model’s predictions.
These include Exact match, CHRF, BLEU and ROGUE-L. Initially, the Exact Match score was mostly
zero, as the model’s full prediction often included additional content beyond the expected
middle segment. To refine this, I modified the Exact Match calculation to compare only the initial
segment of the prediction, truncated to the length of the true middle. This adjustment allowed a
more targeted evaluation, capturing instances where the prediction closely matched the
reference code segment within the middle boundaries.

Additionally, I performed a manual evaluation by labeling each prediction as _correct, partially
correct or incorrect._ The semantics behind these is following:

- _correct:_ The prediction matches the middle code perfectly.
- _partially correct:_ The prediction is partially accurate
- _incorrect:_ The prediction is entirely incorrect.

## Results

All plots related to the evaluation metrics and manual labeling can be found in the
`notebook.ipynb` file. It is important to note that this evaluation was conducted on a very small
dataset of only 30 sample, which should be taken into consideration when interpreting the
results.

There is a certain correlation between the manual evaluation labels and the calculated metrics.
When analyzing predictions labeled as _correct (13.3%)_ the evaluation metrics mostly fell within
the upper range of the scale. Conversely, predictions labeled as _incorrect_ (66.7%) typically aligned
with the lower end of the metric scores, reflecting poorer model performance. For predictions
labeled as _partially correct (20%)_ , the calculated metrics showed intermediate values. This
indicates that while the model generated code that partially resembled the expected completion.

While there is additional data available on prediction accuracy segmented by file, it is not
particularly relevant to draw specific conclusions regarding performance differences across
various code types (e. g., ML CODE, computer vision code) due to the limited size of the dataset.
Such observations would be more meaningful with a larger, more diverse dataset that could
better support analysis by code category.

These observations suggest that there are potential indications of a pattern between the manual
evaluation and the objective metrics used. However, given the limited size of the dataset, these
findings should be interpreted cautiously.


## Conclusion

Findings from this small-scale evaluation indicate that the model’s performance is varied, with
most predictions falling into the _incorrect_ category, highlighting limitations in its ability to
consistently produce accurate completions. The moderate results for partially correct predictions
suggest that the model may sometimes capture relevant context but fails to refine its output
precisely. Only 13.3% of predictions were completely accurate, implying that while the model
shows potential, significant improvements are necessary for it to be reliably used in practical
scenarios. Future work involving a larger dataset would be essential to confirm these initial
patterns and better assess the model’s strengths and areas for improvement.


