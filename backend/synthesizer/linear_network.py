import pandas as pd
import numpy as np
import torch

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn.metrics import precision_recall_fscore_support

from functools import reduce


import spacy
from spacy.matcher import Matcher


from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")

from synthesizer.helpers import expand_working_list
from synthesizer.helpers import get_similar_words
from synthesizer.helpers import soft_match_positives

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "LM"


import pickle
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device=torch.device("cpu" if torch.cuda.is_available() else "cpu")

def get_spanning(matches, sent):
    ranges = [(x[1],x[2]) for x in matches]
    reducer = (lambda acc, el: acc[:-1:] + [(min(*acc[-1], *el), max(*acc[-1], *el))]
    if acc[-1][1] > el[0] else acc + [el] )
    spanning = reduce(reducer, ranges[1::], [ranges[0]])
    
    result_matches = []
    for i, j in spanning:
        result_matches.append([sent.split(" ")[i:j], i, j])
    return result_matches

def check_soft_matching(price, working_list, explain=False, similarity_dict=None, threshold=0.6, topk_on=False, topk=1):
    lemmas = []
    print(working_list)
    for index, distinct_pattern in enumerate(working_list):
        for pattern in distinct_pattern:
            for pat in pattern:
                if 'LEMMA' in pat and 'IN' in pat['LEMMA'] and pat['OP'] == '+':
                    lemmas += pat['LEMMA']['IN']
    lemmas = list(set(lemmas))
    similar_words = dict()
    if len(lemmas) > 0:
        print(lemmas)
        if not similarity_dict is None:
            similar_words = get_similar_words(similarity_dict, price["example"].values, threshold,topk_on=topk_on,topk=topk)
    print(similar_words)

    if len(lemmas) > 0:
        print(lemmas)

        for lemma in lemmas:
            if type(similar_words[lemma]) == type([]): continue
            similar_words[lemma] = [k for k,v in similar_words[lemma].items()]
        for index, patterns in enumerate(working_list):
            for pattern in patterns:
                for pat in pattern:
                    if 'LEMMA' in pat and 'IN' in pat['LEMMA'] and pat['OP'] == '+':
                        if len(pat['LEMMA']['IN']) <= 1 and pat['LEMMA']['IN'][0] in similar_words:
                            pat['LEMMA']['IN'] += similar_words[pat['LEMMA']['IN'][0]]
                            
        return similar_words

def find_similar_words(lemmas, examples, threshold, topk_on=False, topk=1):
    try:
        with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),threshold,topk_on,topk), 'rb') as f:
            similar_words = pickle.load(f)
    except FileNotFoundError:
        similar_words = dict()
    alreadyIn = set()
    for lemma in lemmas:
        if lemma in similar_words:
            alreadyIn.add(lemma)
            continue
        similar_words[lemma] = dict()
    if len(alreadyIn) == len(lemmas):
        return similar_words
    for lemma in lemmas:
        if lemma in alreadyIn: continue
        lemma_embeddings = model.encode(lemma, convert_to_tensor=True)
        for _i, ex in enumerate(examples):
            doc = nlp(str(ex))
            for token in doc:
                #get similarity
                if not str(token.lemma_) in similar_words[lemma] and not str(token.lemma_) in lemmas and not token.is_stop and not token.is_punct and len(token.text.strip(" "))>1:
                    similar_words[lemma][token.lemma_] = util.cos_sim(model.encode(token.lemma_, convert_to_tensor=True), lemma_embeddings)[0][0]

    for lemma in lemmas:
        if lemma in alreadyIn: continue
        if not topk_on:
            similar_words[lemma] = {k:v for k,v in similar_words[lemma].items() if v>threshold}
        else:
            similar_words[lemma] = {k:v for k,v in sorted(similar_words[lemma].items(), key=lambda item: item[1], reverse=True)[:topk]}

    with open('cache/{}/similar_words_examplenum_{}_threshold_{}_topkon_{}_topk_{}.pkl'.format(model_name,len(examples),threshold,topk_on,topk), 'wb') as f:
        pickle.dump(similar_words,f)
    return similar_words

def check_matching(sent, working_list, explain=False):
    matcher = Matcher(nlp.vocab)
    for index, patterns in enumerate(working_list):
        matcher.add(f"rule{index}", [patterns])
    doc = nlp(str(sent))
    
    matches = matcher(doc)
    if(matches is not None and len(matches)>0):
        if(explain):
            return(True, get_spanning(matches, sent) )
        for id, start, end in matches:
            if(str(doc[start:end]).strip() !=""):
                return True
    if(explain):
        return (False, "")
    return False

def patterns_against_examples(file_name, patterns, examples, ids, labels,priority_phrases, similarity_dict=None, soft_threshold=0.6, topk_on=False, topk=1, soft_match_on=False, pattern_customized_dict=None):
    results = []
    for pattern in patterns:
        pattern_result = []
        working_list = expand_working_list(pattern, soft_match_on=soft_match_on, similarity_dict=similarity_dict, pattern_customized_dict=pattern_customized_dict)

        for sent in examples:
            if(check_matching(sent, working_list)):
                pattern_result.append(1)
            else:
                pattern_result.append(0)
        for phrase in priority_phrases:
            print(f'{phrase} with {pattern} => {check_matching(f"{phrase} ", working_list)}')
            if(check_matching(f'{phrase} ', working_list)):
                pattern_result.append(1)
            else:
                pattern_result.append(0)
        
        results.append(pattern_result)
    res = np.asarray(results).T
    df = pd.DataFrame(res, columns=patterns)
    df.insert(0,"sentences", examples+priority_phrases)
    print(df.shape)
    df.insert(0,"labels", labels+([0] * len(priority_phrases)))

    df["id"] = ids+[f'phrase{i}' for i in range(len(priority_phrases))]

    df = df.set_index("id")
    df.to_csv(file_name)
    return df


def train_and_report(patterns, inputs, outputs):
    #Change numpy inputs to tensors 
    outputs = torch.tensor(outputs).reshape(-1,1)
    inputs = torch.tensor(inputs)
    outputs = outputs.to(device)
    inputs = inputs.to(device)

    #train the linear layer for 100 iterations
    #100 chosen at random TODO see what a good number is for iteration

    net = torch.nn.Linear(inputs.shape[1],1, bias=False)
    net = net.to(device)
    sigmoid = torch.nn.Sigmoid()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    losses = []
    net.train()
    for e in range(50):
        optimizer.zero_grad()
        o =  sigmoid.forward(net.forward(inputs.float()))
            
        loss = criterion(o, outputs.float())
            
        losses.append(loss.sum().item())
        loss.backward()
            
        optimizer.step()
    
    pred =  sigmoid.forward(net.forward(inputs.float())).detach().numpy()>0.5
    labeled_prf = precision_recall_fscore_support(outputs, pred, average="weighted")

    fscore = labeled_prf[2]


    return fscore

# define df, columns, true labels
def feature_selector(df):

    positive_examples = df[df['labels']==1]['sentences'].values
    negative_examples = df[df['labels']==0]['sentences'].values

    print(f"==================================Start of Feature Selection===========================================")
    labels = df['labels']
    jj = 0
    ### Controller variables
    patterns_selected = []
    highest_fscore = "0.0"
    df_subset = pd.DataFrame()
    remaining_cols = df.columns.values[4:]


    outputs = df["labels"].values
    while len(patterns_selected)<10 and len(remaining_cols)>0:
        jj += 1
        print(f"Starting iteration {jj} {len(remaining_cols)}")
        #first calculate the fscore
        collector = {}
        local_max_fscore = "0.0"
        for col in remaining_cols:
            col_selected = df[col].astype('int64')
            current_patterns = patterns_selected+[col]
            current_df = pd.concat([df_subset, col_selected], axis=1)
            inputs = current_df.values
            
            # fscore = precision_recall_fscore_support(labels, col_selected,  average="binary")[2]
            fscore = train_and_report(current_patterns, inputs, outputs)
            
                
            exists = str(fscore) in collector
            if(exists):
                collector[str(fscore)].append(col)
                
            else:
                collector[str(fscore)] = [col]

        #sort and get a pattern with high fscore
        selected_starter_pattern = list(collector.values())[-1]
        collector = {k: v for k, v in sorted(collector.items(), key=lambda item: item[0])}
        current_fscore = list(collector.keys())[-1]

        if(current_fscore>highest_fscore):
            highest_fscore = current_fscore
        else:
            break
        selected_starter_pattern = list(collector.values())[-1]

        #Group the correlated ones and pick the shortest
        rowss = df[selected_starter_pattern]
        correlation = rowss.corr()
        correlation.loc[:,:] =  np.tril(correlation, k=-1)
        cor = correlation.stack()
        ones = cor[cor >=0.8].reset_index().loc[:,['level_0','level_1']]
        ones = ones.query('level_0 not in level_1')
        grps = list(ones.groupby('level_0').groups.keys())
        colls = []
        for i in grps:
            groups = ones[ones["level_0"]==i].values
            set_maker = []
            for patterns in groups:
                set_maker += patterns.tolist()
            colls.append(sorted(set_maker, key=len)[0])
            
        for selected_starter_pattern in colls:
            patterns_selected.append(selected_starter_pattern)
            df_subset[selected_starter_pattern] = df[selected_starter_pattern].astype('int64')
            try:
                selected_starter_series = df[selected_starter_pattern][0]
                
                corr = df.corr()
                to_drop = [c for c in corr.columns if corr[selected_starter_pattern][c] >= 0.8]
                df = df.drop(to_drop, axis=1)

                #create a new df with combination of current one
                remaining_cols = df.columns.values[4:]
                for collumn in remaining_cols:
                    df[collumn] = np.logical_or(df[collumn], selected_starter_series)
            except:
                print("We already removed ", selected_starter_pattern)
            for coll in remaining_cols:
                df[coll] = np.logical_or(df[coll], selected_starter_series)
        
        print(f"Finishing iteration {jj} {len(remaining_cols)}, --- {patterns_selected}, {highest_fscore}")
    
    print(f"---------------------------Summary---------------------------")
    print(f"Patterns {patterns_selected}")
    print(f"Positive examples \n{positive_examples}")
    print(f"Negative examples \n{negative_examples}")

    print(f"==================================End of Feature Selection===========================================")
    return patterns_selected



def feature_selector_2(df, k,  deleted_patterns=[], pinned_patterns=[]):

    patterns = []
    all_cols = df.columns.values
    if(len(deleted_patterns)>0):
        df = df.drop(labels=deleted_patterns, axis=1, errors='ignore')


    outs = df["labels"].values
    to_delete = np.array(deleted_patterns)

    for i in range(k-len(pinned_patterns)):

        inputs = df.iloc[:,3:].values

        selector = SelectKBest(f_classif, k=1)
        X_new = selector.fit_transform(inputs, outs)

        cols = selector.get_support(indices=True)
        selected_patterns = np.take(df.columns.values,[x+3 for x in cols] )


        #get rid of all features correlated 
        corr = df.corr()
        corr.head()
        
        to_drop = [c for c in corr.columns if corr[selected_patterns[0]][c] >= 0.5  or pd.isnull(corr[selected_patterns[0]][c])] #0.8 chosen at random

        df = df.drop(to_drop, axis=1)
        print(f'picked {selected_patterns[0]}')

        
        patterns.append(selected_patterns[0])
        if(df.shape[1]<=4):
            break
    if(len(pinned_patterns)>0):
        for pat in pinned_patterns:
            print("pinned ", pat)
            if pat in all_cols:
                patterns.append(pat)
    
    return patterns

def train_linear_mode(df, data, theme, soft_match_on=True, words_dict=None, similarity_dict=None, soft_threshold=0.6, soft_topk_on=False, 
soft_topk=1, pattern_customized_dict=None, deleted_patterns=[], pinned_patterns=[]):
    print("pinned patterns ", pinned_patterns)
    print("deleted patterns ", deleted_patterns)
    outs = df["labels"].values

    cols = feature_selector_2(df, 5, deleted_patterns, pinned_patterns)

    smaller_inputs =  df[cols].values


    ins = torch.tensor(smaller_inputs, dtype=torch.int64)
    ins = ins.to(device)
    

    output = torch.tensor(outs, dtype=torch.int64).reshape(-1,1)
    output=output.to(device)


    net = torch.nn.Linear(ins.shape[1],1, bias=False)
    net=net.to(device)
    sigmoid = torch.nn.Sigmoid()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    losses = []
    net.train()
    print("training ...")
    for e in range(100):
        optimizer.zero_grad()
        o =  sigmoid.forward(net.forward(ins.float()))
            
        loss = criterion(o, output.float())
            
        losses.append(loss.sum().item())
        loss.backward()
            
        optimizer.step()
    

    ins=ins.cpu()
    output=output.cpu()
    net=net.cpu()
    


    pred =  sigmoid.forward(net.forward(ins.float())).detach().numpy()>0.5

    labeled_prf = precision_recall_fscore_support(outs, pred, average="binary")

    selected_patterns = cols
    selected_working_list = []
    matched_parts = {}
    for pattern in selected_patterns:
        selected_working_list.append(expand_working_list(pattern, soft_match_on=soft_match_on, similarity_dict=similarity_dict, pattern_customized_dict=pattern_customized_dict))

    for pattern in selected_patterns:
        matched_parts[pattern] = {}

    running_result = []

    for sentence,id in zip(data["example"].values, data["id"].values):
        temp = []
        
        for i in range(len(selected_working_list)):
            it_matched = check_matching(sentence, selected_working_list[i], explain=True)

            temp.append(int(it_matched[0]))
            matched_parts[selected_patterns[i]][id] = it_matched[1] 
            


        running_result.append(temp)

    

    entire_dataset_ins = torch.Tensor(running_result)

    try:
        entire_dataset_outs = torch.Tensor(data[theme].values).reshape(-1,1)
    except:
        entire_dataset_outs = torch.Tensor([0]*len(data['id'].values.tolist())).reshape(-1,1)
    

    print(entire_dataset_ins.shape)

    overall_prob = sigmoid.forward(net.forward(entire_dataset_ins.float())) 

    overall_pred = overall_prob.detach().numpy()>0.5
    ids = data['id'].values.tolist()

    overall_prf = precision_recall_fscore_support(entire_dataset_outs, overall_pred, average="binary")

    response = dict()

    response["explanation"] = matched_parts


    patterns =[]

    weights = net.weight.detach().numpy()[0].tolist()
    for i in range(len(cols)):
        temp = dict()
        prf = precision_recall_fscore_support(output, df[ cols[i]], average="binary" )
        temp["pattern"] = selected_patterns[i]
        temp["precision"] = prf[0]
        temp["recall"] = prf[1]
        temp["fscore"] = prf[2]
        temp["weight"] = weights[i]
        temp["status"] = 1

        patterns.append(temp)

    response["fscore"] = labeled_prf[2]
    response["recall"] = labeled_prf[1]
    response["precision"] = labeled_prf[0]


    response["overall_fscore"] = overall_prf[2]
    response["overall_recall"] = overall_prf[1]
    response["overall_precision"] = overall_prf[0]


    response["patterns"] = patterns
    response["weights"] = net.weight.detach().numpy()[0].tolist()



    response["scores"] = { x:y[0] for x,y in zip(ids, overall_prob.tolist()) }

    return response
