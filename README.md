
The Social Media Mining for Health Applications (#SMM4H) Shared Task involves natural language processing (NLP) challenges of using social media data for health research, including informal, colloquial expressions and misspellings of clinical concepts, noise, data sparsity, ambiguity, and multilingual posts. For each of the five tasks below, participating teams will be provided with a set of annotated tweets for developing systems, followed by a three-day window during which they will run their systems on unlabeled test data and upload the predictions of their systems to CodaLab. Informlsation about registration, data access, paper submissions, and presentations can be found below.

Task 1: Automatic classification of tweets that mention medications

This binary classification task involves distinguishing tweets that mention a medication or dietary supplement (annotated as “1”) from those that do not (annotated as “0”). For this task we follow the definition of a drug product and of a dietary supplement as stated by the FDA. These definitions and concrete examples can be found in the guidelines we followed during the annotation of the data for the Task 1. In 2018, this task was organized using a data set that contained an artificially balanced distribution of the two classes. This year, the data set1 represents the natural, highly imbalanced distribution of the two classes among tweets posted by 112 women during pregnancy2, with only approximately 0.2% of the tweets mentioning a medication. Training and evaluating classifiers on this year’s data set will more closely model the detection of tweets that mention medications in practice.

Training data: 69,272 (181 “positive” tweets; 69,091 “negative” tweets)
Test data: 29,687 tweets
Evaluation metric: F1-score for the “positive” class (i.e., tweets that mention medications)
Contact information: Davy Weissenbacher (dweissen@pennmedicine.upenn.edu)
Task 2: Automatic classification of multilingual tweets that report adverse effects

This binary classification task involves distinguishing tweets that report an adverse effect (AE) of a medication (annotated as “1”) from those that do not (annotated as “0”), taking into account subtle linguistic variations between AEs and indications (i.e., the reason for using the medication). This classification task has been organized for every past #SMM4H Shared Task, but only for tweets posted in English. This year, this task also includes distinct sets of tweets posted in French and Russian.

English
Training data: 25,672 tweets (2,374 “positive” tweets; 23,298 “negative” tweets)
Test data: ~5,000 tweets
Contact information: Arjun Magge (amaggera@asu.edu)
French
Training data: 2,426 tweets (39 “positive” tweets; 2,387 “negative” tweets)
Test data: 607 tweets
Contact information: Anne-Lyse Minard (anne-lyse.minard@univ-orleans.fr)
Russian
Training data: 7,612 tweets (666 “positive” tweets; 6,946 “negative” tweets)
Test data: 1,903 tweets
Contact information: Elena Tutubalina (tutubalinaev@gmail.com)
We thank Yandex.Toloka for supporting the shared task and providing credits for data annotation in Russian.
Evaluation metric: F1-score for the “positive” class (i.e., tweets that report AEs)
Task 3: Automatic extraction and normalization of adverse effects in English tweets

This task, organized for the first time in 2019, is an end-to-end task that involves extracting the span of text containing an adverse effect (AE) of a medication from tweets that report an AE, and then mapping the extracted AE to a standard concept ID in the MedDRA vocabulary (preferred terms). The training data includes tweets that report an AE (annotated as “1”) and those that do not (annotated as “0”). For each tweet that reports an AE, the training data contains the span of text containing the AE, the character offsets of that span of text, and the MedDRA ID of the AE. For some of the tweets that do not report an AE, the training data contains the span of text containing an indication (i.e., the reason for using the medication) and the character offsets of that span of text, allowing participants to develop techniques for disambiguating AEs and indications.

Training data: 2,376 tweets (1,212 “positive” tweets; 1,155 “negative” tweets)
Test data: ~1,000 tweets
Evaluation metric: F1-score for the “positive” class (i.e., the correct AE spans and MedDRA IDs for tweets that report AEs)
Contact information: Arjun Magge (amaggera@asu.edu)
Task 4: Automatic characterization of chatter related to prescription medication abuse in tweets

This new, multi-class classification task involves distinguishing, among tweets that mention at least one prescription opioid, benzodiazepine, atypical anti-psychotic, central nervous system stimulant or GABA analogue, tweets that report potential abuse/misuse (annotated as “A”) from those that report non-abuse/-misuse consumption (annotated as “C”), merely mention the medication (annotated as “M”), or are unrelated (annotated as “U”)3.

Training data: 13,172 tweets
Test data: 3,271 tweets
Evaluation metric: F1-score for the “potential abuse/misuse” class
Contact information: Abeed Sarker (abeed@dbmi.emory.edu)
Task 5: Automatic classification of tweets reporting a birth defect pregnancy outcome

This new, multi-class classification task involves distinguishing three classes of tweets that mention birth defects: “defect” tweets refer to the user’s child and indicate that he/she has the birth defect mentioned in the tweet (annotated as “1”); “possible defect” tweets are ambiguous about whether someone is the user’s child and/or has the birth defect mentioned in the tweet (annotated as “2”); “non-defect” tweets merely mention birth defects (annotated as “3”)4,5.

Training data: 18,397 tweets (953 “defect” tweets; 956 “possible defect” tweets; 16,488 “non-defect” tweets)
Test data: 4,602 tweets
Evaluation metric: micro-averaged F1-score for the “defect” and “possible defect” classes
Contact information: Ari Klein (ariklein@pennmedicine.upenn.edu)
Important Dates

Training data available: January 15, 2020 (may be sooner for some tasks)
Test data available: April 2, 2020
System predictions for test data due: April 5, 2020 (23:59 CodaLab server time)
System description paper submission deadline: May 5, 2020
Notification of acceptance of system description papers: June 10, 2020
Camera-ready papers due: June 30, 2020
Workshop: September 13, 2020

* All deadlines, except for system predictions (see above), are 23:59 UTC (“anywhere on Earth”).