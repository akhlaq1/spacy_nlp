from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import urllib.request
import json
import spacy
from spacy.matcher import Matcher
from spacy import displacy

app = Flask(__name__)

## LOAD SPACY 
nlp = spacy.load('en_core_web_sm')

# LOAD TEXT DATA
url = r'https://www.dropbox.com/s/6a1dgvyg04qztx5/df100.csv?dl=1'
data = urllib.request.urlopen(url)
df = pd.read_csv(data)
data.close()
# print('data loaded: ', df.shape)
data_dict = df.iloc[:,1].to_dict() 

## LOAD PATTERNS TO MATCH
url = r'https://www.dropbox.com/s/9ozpb1bc4gqcd4h/hcat_v010.jsonl?dl=1'
data = urllib.request.urlopen(url)
data2 = data.readlines()
patterns = []
for line in data2:
    patterns.append(json.loads(line))
data.close()
print('patterns loaded: ', len(patterns))

@app.route('/', methods=['GET'])
def main_route():
    return "<h1>App worked fine</h1>"

@app.route('/test', methods=['POST','GET'])
def hello():
    return render_template('hello.html', content = data_dict )

@app.route('/output', methods=['POST','GET'])
def index():
    ## FUNCTION TO ADD PATTERNS TO THE MATCHER
    def addPatternsToMatcher(patterns):
        # ADD PATTERNS TO THE MATCHER
        labels = list(set([x['LABEL'] for x in patterns]))
        for label in labels:
            # check label is in the vocab
            if label not in nlp.vocab: # if the label is not in the vocab 
                #print(f'Adding {label} to the vocab')
                lex = nlp.vocab[label]        # then add the label to the vocab
                assert label in nlp.vocab, f'tried to add {label} to nlp.vocab, but failed!'
            
            # add list of patterns to the matcher
            label_patterns = [x['PATTERN'] for x in patterns if x['LABEL'] == label]
            matcher.add(label, None, *label_patterns)# the '*' is needed for list of patterns
        return



    ## FUNCTION TO REMOVE OVERLAPPING/NON-GREEDY MATCHES
    def removeNonGreedyMatches(matches):
        for match_id1, start1, end1 in matches:
            for match_id2, start2, end2 in matches:
                if match_id1 == match_id2 and ((start1 <= start2 and end1 > end2) or (start1 < start2 and end1 >= end2)):
                    tuple_to_remove = (match_id2, start2, end2)
                    matches.pop(matches.index(tuple_to_remove))
        return matches 


    ## FUNCTION TO GET MATCHES
    def getMatches(post_id):
        post_text = request.form['text1']
        doc = nlp(post_text)
        # apply the matcher
        matches = matcher(doc)
        # remove overlapping matches
        matches = removeNonGreedyMatches(matches)
        return doc, matches

    ## FUNCTION TO FORMAT DISPLACY MATCHES
    def formatMatchesForDisplacy(doc, matches):
        # convert match object into displacy match object (using character, not word, position)
        displacy_matches = []
        match_ents = []
        match_labels = []
        for match in matches:
            #print(match)
            match_id, start, end = match
            match_label = nlp.vocab.strings[match_id]
            match_labels.append(match_label)
            span = doc[start:end]
            match_ent = {'start': span.start_char,
                        'end': span.end_char,
                        'label': nlp.vocab.strings[match_id]}
            match_ents.append(match_ent)
        displacy_matches.append({'text': doc.text, 'ents': match_ents, 'title': None, 'settings':{}})
        labels_extracted = displacy_matches[0]['ents']
        labels_list = []
        for l in labels_extracted:
            labels_list.append(l['label'])
        return displacy_matches, labels_list #, match_labels


    ## FUNCTION TO DISPLAY MATCHES
    def displayMatches(displacy_matches, labels_list):

        # get a list of the labels
        labels_set = set([nlp.vocab.strings[match[0]] for match in matches])
        labels = list(labels_set)
        labels.sort()
        
        # define the colors for the label types
        colors ={}
        for i in range(0, len(labels)):
        
                # harm = shades of red
                if 'harm_0' in labels[i]:
                    colors[labels[i].upper()]='#ffe6e6'
                elif 'harm_1' in labels[i]:
                    colors[labels[i].upper()]='#ff9999'
                elif 'harm_2' in labels[i]:
                    colors[labels[i].upper()]='#ff6666' 
                elif 'harm_3' in labels[i]:
                    colors[labels[i].upper()]='#ff3333' 
                elif 'harm_4' in labels[i]:
                    colors[labels[i].upper()]='#e60000'
                    
                # stages and uncategorised (x) = dark grey
                elif 'stg_' in labels[i]:
                    colors[labels[i].upper()]= '#E9E9E9'
                elif 'x_' in labels[i]:
                    colors[labels[i].upper()]= '#E9E9E9'
                
                # staff = light grey
                elif 'staff_' in labels[i]:
                    colors[labels[i].upper()]= '#F4F4F4'
                    
                # positive = green
                elif 'pos_' in labels[i]:
                    colors[labels[i].upper()] = '#7fbf7b'
                
                # negative = purple
                elif 'neg_' in labels[i]:
                    colors[labels[i].upper()] = '#af8dc3'
                    
                # missing color = black
                else:
                    colors[labels[i].upper()]= '#000000' # black
            
        options = {"ents": [label.upper() for label in labels], "colors":colors}
        
        # display
        ex = [{"text": "But Google is starting from behind.",
        "ents": [{"start": 4, "end": 10, "label": "ORG"}],
        "title": None}]
        html1 = displacy.render(displacy_matches, style='ent', page=True, manual=True, options=options)
        return html1

    ## DISPLAY MATCHES FOR TEXT
    text_id = 35
    matcher = Matcher(nlp.vocab) # initialize/reset the matcher
    addPatternsToMatcher(patterns)
    doc, matches = getMatches(text_id)
    displacy_matches, labels_list = formatMatchesForDisplacy(doc, matches)
    data = jsonify(displacy_matches)
    return displayMatches(displacy_matches, labels_list)

if __name__ == "__main__":
    app.run()    


