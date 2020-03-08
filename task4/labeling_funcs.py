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
def lf_contains_seeking(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'looking for' in text or 'need' in text or 'seeking' in text or 'found' in text or 'find' in text or 'get' or 'want' in text:
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
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'snort' in text or 'snorting' in text or 'snorted' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_wanting(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'wish i had' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_too_much(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'too much' in text or 'overdose' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_mixing(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'mix' in text or 'mixing' in text or 'mixed' in text or 'mixin' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_drug_and_other_substance(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'alcohol' in text or 'weed' in text or 'marijuana' in text:
        return a
    else:
        return ABSTAIN

@labeling_function()
def lf_contains_popping(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'pop' in text or 'popped' in text or 'poppin' in text or 'popping' in text:
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

@labeling_function()
def lf_contains_steal(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'steal' in text or 'stole' in text or 'stolen' in text:
        return a
    else:
        return ABSTAIN

labeling_function()
def lf_contains_(x):
    text = x.lower()
    if set(text.split()).intersection(all_drugs) and 'mix' in text or 'mixing' in text or 'mixed' in text or 'mixin' in text:
        return a
    else:
        return ABSTAIN

