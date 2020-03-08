from snorkel.labeling import labeling_function

a = 1
ABSTAIN = -1
m = 0
c = 2

school_words = ['studying', 'study', 'homework', 'school', 'library', 'education', 'class', 'semester', 'exam', 'test', 'paper', 'campus', 'essay', 'college', 'university']
cleaning_words = ['clean', 'cleaned', 'cleaning', 'tidy', 'tidying']

all_drugs = []

@labeling_function()
def lf_contains_adderall_and_school(x):
    # Return a label of a if mention of  "adderall" in text and there is a school word in the text, otherwise ABSTAIN
    text = x.lower()
    if "adderall" in text and set(text.split()).intersection(set(school_words)):
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_adderall_and_cleaning(x):
    # Return a label of a if mention of  "adderall" in text and there is a cleaning word in the text, otherwise ABSTAIN
    text = x.lower()
    if "adderall" in text and set(text.split()).intersection(set(cleaning_words)):
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_for_sale(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'for sale' in text or 'selling' in text or 'sell' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_buying(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'buy' in text or 'buying' in text or 'bought' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_my_fix(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'my fix' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_snort(x):
    ########### THIS ONE NEEDS TO BE MODIFIED TO DISTINGUISH BETWEEN WHETHER THE SUBJECT SNORTING IS THE TWEETER OR SOMEONE ELSE BECAUSE IT ONLY APPLIES IF IT'S THE TWEETER ##############
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'snort' in text or 'snorting' in text or 'snorted' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_do_a_line(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and ('do a' in text or 'does a' in text or 'does another' in text or 'do another' in text or 'take a' in text) and 'line' in text:
        return a
    else:
        return ABSTAIN
