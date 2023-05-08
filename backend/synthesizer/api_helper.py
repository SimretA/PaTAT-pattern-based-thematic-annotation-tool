from tokenize import group
from synthesizer.linear_network import patterns_against_examples, train_linear_mode
from synthesizer.penality_based_threaded import Synthesizer
from synthesizer.helpers import dict_hash
from synthesizer.helpers import get_patterns
from synthesizer.helpers import get_similarity_dict
from synthesizer.helpers import expand_working_list
from synthesizer.helpers import NN_cluster
from synthesizer.helpers import NN_classificaion
from synthesizer.helpers import NN_multi_classificaion
from synthesizer.helpers import pattern_clusters
from synthesizer.cache_helper import RepeatedTimer

import pandas as pd
import json
import spacy
import random
from time import sleep


import asyncio

loop = asyncio.get_event_loop()

nlp = spacy.load("en_core_web_sm")

SEED = 42
random.seed(SEED)
#####################
# Pattern status 
# 0 - not selected
# 1 - selected
# 2 - deleted
#       
####################
class APIHelper:
    
    def __init__(self, user="test"):

        self.name = user

        self.positive_examples_collector = {}
        self.negative_examples_collector = {}

        self.negative_phrases = []
        # self.theme = "ethos_400_1"
        self.theme = "yelp"
        # self.theme = self.get_dataset_for_part(user.split("_")[0], user.split("_")[1])
        self.selected_theme =  None #"price"
        



        self.data = pd.read_csv(f"data/{self.theme}.csv", delimiter=",")

        self.positive_phrases = []
        self.negative_phrases = []

        self.labels = {}
        self.themes = {}
        self.results = {}

        self.pattern_customized_dict = {}
        self.soft_match_on = True
        self.only_soft_match = False
        self.words_dict = {}
        self.similarity_dict = {}
        self.soft_threshold = 0.6
        self.soft_topk_on = False
        self.topk = 1
        self.words_dict, self.similarity_dict = get_similarity_dict(self.data["example"].values, soft_threshold=self.soft_threshold, file_name=self.theme)
        # print(list(self.similarity_dict['pricey'].keys()))


        self.all_themes = self.get_themes()
        self.element_to_label = {}
        self.theme_to_element = {}
        self.theme_to_negative_element = {} #Used to collect examples labeled no in binary mode
        self.element_to_sentence = {}
        self.synthesizer_collector = {}
        self.initialize_synthesizers(self.get_themes())
        self.binary_mode = False

        self.restore_stash()


        # self.rt = RepeatedTimer(100, self.stash_stuff)

    def get_dataset_for_part(self, participant, condition):
        file_name = ""
        exp_design = pd.read_csv("exp_plan/participat_to_condition.csv")
        try:
            data = exp_design[exp_design["participantid"] == participant].values[0][-2]
            file_name = data
            if(condition=="man"):
                file_name += "_1"
            
            if(condition=="bert"):
                file_name += "_2"
            
            if(condition=="pata"):
                file_name += "_3"
            print("the file is", file_name)
            return file_name
        except Exception as e:
            print(e)


    def restore_stash(self):
        try:
            
            # JSON file
            f = open(f'user_checkpoints/{self.name}.json', "r")
            
            # Reading from file
            data = json.loads(f.read())
            print(data)

            #TODO
            #Restore all of the values here
            self.theme_to_element = data["theme_to_element"]
            self.theme_to_negative_element = data["theme_to_negative_element"]
            self.all_themes = data["all_themes"]
            self.binary_mode = data["binary_mode"]
            self.selected_theme = data["selected_theme"]
            self.element_to_label = data["element_to_label"]
            self.element_to_sentence = {key:nlp(value) for key, value in data["element_to_sentence"].items()}
            
            if(data["all_themes"] is not None):
                self.initialize_synthesizers(data["all_themes"], restored=True, restored_data=data["synthesizer_collector"])
            
        except Exception as e:
            print("Unable to get stashed data")
            print(e)
            
    
    def stash_stuff(self):
        
        if len(list(self.theme_to_element.keys())) == 0:
            # print("Nothing to cache yet")
            return
        # print("Caching", len(list(self.theme_to_element.keys())))

        #THINGS TO CACHE
        cache = {
            "theme_to_element":self.theme_to_element,
            "theme_to_negative_element":self.theme_to_negative_element,
            "all_themes":self.all_themes,
            "binary_mode": self.binary_mode,
            "selected_theme":self.selected_theme,
            "element_to_label":self.element_to_label,
            "element_to_sentence":{key:str(value) for key, value in self.element_to_sentence.items()},
            "synthesizer_collector":{theme:{"results":self.synthesizer_collector[theme].results, 
                            "deleted_patterns":self.synthesizer_collector[theme].deleted_patterns, 
                            "pinned_patterns":self.synthesizer_collector[theme].pinned_patterns,
                            "df_tracker": {} if self.synthesizer_collector[theme].df_tracker is None else self.synthesizer_collector[theme].df_tracker.to_json()} for theme in self.synthesizer_collector.keys()}
        }

        with open(f'user_checkpoints/{self.name}.json', 'w', encoding ='utf8') as json_file:
            json.dump(cache, json_file, allow_nan=True)
    
    
    def save_cache(self, pattern_set, positive_examples= None, negative_examples=None, pos_ids=None, neg_ids=None):
        file_name = f"{self.selected_theme}_{dict_hash(self.labels)}" #TODO Add user session ID

        
        # examples = list(self.positive_examples_collector.values())+list(self.negative_examples_collector.values())
        # ids = list(self.positive_examples_collector.keys())+list(self.negative_examples_collector.keys())
        # labels = [self.labels[x] for x in ids]

        examples = positive_examples+negative_examples
        ids = pos_ids+neg_ids
        labels = [1]*len(pos_ids) + [0]*len(neg_ids)

        df = patterns_against_examples(file_name=f"cache/{file_name}.csv",patterns=list(pattern_set.keys()), examples=examples, ids=ids, labels=labels, priority_phrases=self.negative_phrases, soft_match_on=self.soft_match_on, similarity_dict=self.similarity_dict, soft_threshold=self.soft_threshold, pattern_customized_dict=self.pattern_customized_dict)
        
        return df

    def ran_cache(self):
        file_name = f"{self.selected_theme}_{dict_hash(self.labels)}" 
        try:
            df = pd.read_csv(f"cache/{file_name}.csv")
            return df
        except:
            print("cache miss")
            return None
    ####### End Points ######

    def set_theme(self, theme):
        self.selected_theme = theme
        try:
            self.data['positive'] = self.data[self.selected_theme]
        except:
            print("new theme added")
        self.stash_stuff()
        return self.get_labeled_dataset()
    
    def get_themes(self):
        return []
        return list(self.data.columns.unique())[2:-1]
    
    def get_selected_theme(self):
        return self.selected_theme


    def label_by_phrase(self, phrase, label, positive, elementId):
        if(positive==0):
            print("Before", self.negative_phrases)
            self.negative_phrases.append(nlp(phrase.strip()))
            print("After", self.negative_phrases)
        elif positive==1:
            self.positive_phrases.append(nlp(phrase.strip()))
        print("positive phrases",self.positive_phrases, "negative phrases",self.negative_phrases)
        self.stash_stuff()
        return {"status":200, "message":"ok", "phrase":phrase, "label":label, "positive":positive}


    def add_theme(self, theme):
        if(theme not in self.all_themes):
            self.themes[theme] = {}
            self.all_themes.append(theme)
            self.synthesizer_collector[theme] = Synthesizer(positive_examples = [], negative_examples = [], soft_match_on=self.soft_match_on, words_dict=self.words_dict, similarity_dict=self.similarity_dict,
                soft_threshold=self.soft_threshold, pattern_customized_dict=self.pattern_customized_dict)
            self.theme_to_element[theme] = []
            self.theme_to_negative_element[theme] = []


        self.stash_stuff()
        return list(self.all_themes)



    def batch_label(self, id, label):
        #check if label already exisits in the oposite collector and remove if it does
        exists = id in self.labels
        if(exists):
            previous_label = self.labels[id]
            if(previous_label==0):
                # remove from negative_example_collectore
                del self.negative_examples_collector[id]
            else:
                # remove from positive_example_collectore
                del self.positive_examples_collector[id]
        
        self.labels[id] = label
        sentence = nlp(self.data[self.data["id"] == id]["example"].values[0])
        if(label==0):
            self.negative_examples_collector[id] = sentence
        elif label==1:
            self.positive_examples_collector[id] = sentence
        
        self.stash_stuff()
        return {"status":200, "message":"ok", "id":id, "label":label}
    
    def clear_label(self):
        self.labels.clear()
        
        self.negative_examples_collector.clear()
        self.positive_examples_collector.clear()
        self.negative_phrases = []

        self.stash_stuff()
        return {"message":"okay", "status":200}



    def get_labeled_dataset(self):
        dataset = []

        ids = self.data["id"].values
        for i in ids:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0].capitalize()

            try:
                item["true_label"] = self.data[self.data["id"] == i][self.selected_theme].values.tolist()[0] if self.selected_theme in self.data.columns else None
            except:
                item["true_label"] = None
            item["score"] = None

            if(str(i) in self.element_to_label):
            # if(str(i) in self.labels):
                item["user_label"] = []
                item["negative_user_label"] = []
                for label in self.element_to_label[str(i)]:
                    if label in self.theme_to_element and str(i) in self.theme_to_element[label]:
                        item["user_label"].append(label)
                    if label in self.theme_to_negative_element and str(i) in self.theme_to_negative_element[label]:
                        item["negative_user_label"].append(label)
            else:
                item["user_label"] = None
                item["negative_user_label"] =  None

            # if(str(i) in self.element_to_label):
            # # if(str(i) in self.labels):
            #     item["user_label"] = self.element_to_label[str(i)]
            # else:
            #     item["user_label"] = None
            # print()

            dataset.append(item)


        return dataset
    
    def get_related(self, id):

        score= self.synthesizer_collector[self.selected_theme].results['scores'][id] 
        explanation = self.synthesizer_collector[self.selected_theme].results['explanation']
        this_pattern_matches = []
        for key, value in explanation.items():
            if(value[id]!=""):
                this_pattern_matches.append(key)
        print("match related with ", this_pattern_matches)
        

        related = []

        related_highlights = {}

        for sentence_id in list(self.data['id'].values):
            if sentence_id == id:
                continue
            if self.synthesizer_collector[self.selected_theme].results['scores'][sentence_id] == score:
                related.append(sentence_id)
                related_highlights[sentence_id] = []
                for pattern in this_pattern_matches:
                    hglgt =  " ".join(explanation[pattern][sentence_id][0][0])
                    # print("related highlights ", explanation[pattern][sentence_id])
                    related_highlights[sentence_id].append(hglgt)
        # print("highlight ", related_highlights)





        
        dataset = []
        
        for i in related:
            item = dict()
            item["id"] = str(i)
            item["example"] = self.data[self.data["id"] == i]["example"].values[0]
            item["score"] = None
            if(str(i) in self.labels):
                item["user_label"] = self.labels[str(i)]
            else:
                item["user_label"] = None

            dataset.append(item)


        return dataset, related_highlights


    def explain_pattern(self, pattern):
        
        exp = {}
        pattern = pattern.replace('+', ', ')
        pattern = pattern.replace('|', ', ')
        pattern_list = pattern.split(", ")
        for pat in pattern_list:
            pattern_expanded = expand_working_list(pat, soft_match_on=True, similarity_dict=self.similarity_dict, pattern_customized_dict=self.pattern_customized_dict)[0][0]
            print("pattern expanded is ", pattern_expanded)
            if('LEMMA' in pattern_expanded and pat[0]=='('):
                print(pattern_expanded['LEMMA'])
                exp[pat] = [True, pattern_expanded['LEMMA']["IN"]]
            else:
                 exp[pat] = [False, list(pattern_expanded.keys())]
        return exp

        


######################################################################################################################################################

    def initialize_synthesizers(self, themes, restored=False, restored_data=None):
        for theme in themes:
            print("Intializing ", theme)
            self.synthesizer_collector[theme] = Synthesizer(positive_examples = [], negative_examples = [], soft_match_on=self.soft_match_on, words_dict=self.words_dict, similarity_dict=self.similarity_dict,
            soft_threshold=self.soft_threshold, pattern_customized_dict=self.pattern_customized_dict)
            if not restored:
                self.theme_to_element[theme] = []
                self.theme_to_negative_element[theme] = []
            
            if restored:
                print("theme ", theme, "Deleted ", restored_data[theme]["deleted_patterns"])
                self.synthesizer_collector[theme].deleted_patterns = restored_data[theme]["deleted_patterns"]
                self.synthesizer_collector[theme].pinned_patterns = restored_data[theme]["pinned_patterns"]
                # self.synthesizer_collector[theme].df_tracker = pd.read_json(restored_data[theme]["df_tracker"])


    
    def merge_themes(self, theme1, theme2, new_theme):
        # if(theme1 not in self.theme_to_element or theme2 not in self.theme_to_element):
        # self.selected_theme= new_theme
        self.add_theme(new_theme)
        # self.all_themes.append(new_theme)
        ids_to_updae = []
        

        self.synthesizer_collector[new_theme] = Synthesizer(positive_examples = [], negative_examples = [], soft_match_on=self.soft_match_on, words_dict=self.words_dict, similarity_dict=self.similarity_dict,
            soft_threshold=self.soft_threshold, pattern_customized_dict=self.pattern_customized_dict)
        self.theme_to_element[new_theme] = []
        self.theme_to_negative_element[new_theme] = []
        #For each example in theme1 and theme2 add them to new_theme
        for elementId in self.theme_to_element[theme1]:
            self.theme_to_element[new_theme].append(elementId)
            self.element_to_label[elementId].append(new_theme)
            ids_to_updae.append(elementId)
        for elementId in self.theme_to_element[theme2]:
            ids_to_updae.append(elementId)
            self.theme_to_element[new_theme].append(elementId)
            if(new_theme not in self.element_to_label):
                self.element_to_label[elementId].append(new_theme)
            
        
        # for elementId in self.theme_to_negative_element[theme1]:
        #     self.theme_to_negative_element[new_theme].append(elementId)
            
        # for elementId in self.theme_to_negative_element[theme2]:
        #     self.theme_to_element[new_theme].append(elementId)
            


        self.stash_stuff()

        return {"selected_theme":self.selected_theme, "new_theme":new_theme, "all_themes":self.all_themes, "pos_update":list(set(ids_to_updae))}
        # print(f"merging these themes {theme1} {theme2}, {self.theme_to_element[new_theme]}")
    
    def delete_theme(self, theme):
        pos_ids = []
        neg_ids = []
        if(theme in self.all_themes):
            index = self.all_themes.index(theme)
            pos_ids = self.theme_to_element[theme][:]
            neg_ids = self.theme_to_negative_element[theme][:]
            self.all_themes.remove(theme)

            if len(self.all_themes)>0:
                next = (index+1)%len(self.all_themes)
                self.selected_theme = self.all_themes[next]
            else:
                self.selected_theme=None
        for elementId in pos_ids:
            if(theme in self.element_to_label[elementId]):
                self.element_to_label[elementId].remove(theme)
        
        del self.theme_to_element[theme]
        del self.theme_to_negative_element[theme]
        self.stash_stuff()

        return {"selected_theme":self.selected_theme, "all_themes":self.all_themes, "deleted_theme":theme, "pos_update":pos_ids, "neg_update": neg_ids}
    
    def get_examples_by_patterns(self, theme, patterns, new_theme_name):
        group1 = {}
        group2 = {}
        #get results/explanation for original theme
        explanation = self.synthesizer_collector[theme].results["explanation"]

        for elementId in self.theme_to_element[theme]:
            matched = False
            for pattern in patterns:
                matched_sentences = explanation[pattern]

                if(type(matched_sentences[elementId])==type(list())):
                    group2[elementId] = [str(self.element_to_sentence[elementId]), 1]
                    matched = True
                    continue
            if(not matched):
                group1[elementId] =  [str(self.element_to_sentence[elementId]), 1]

        res  = dict()

        res['group1'] = group1
        res["group2"] = group2

        return res
                
    def rename_theme(self, theme, new_name):
        try:
            # self.all_themes = self.get_themes()
            self.all_themes = list(map(lambda x: x.replace(theme,  new_name), self.all_themes))
            self.theme_to_element[new_name] = self.theme_to_element[theme]
            self.theme_to_negative_element[new_name] = self.theme_to_negative_element[theme]
            self.synthesizer_collector[new_name] = self.synthesizer_collector[theme]
            if(self.selected_theme==theme):
                self.selected_theme = new_name

            for elementId in self.theme_to_element[new_name]:
                self.element_to_label[elementId] = list(map(lambda x: x.replace(theme, new_name), self.element_to_label[elementId]))
            
            for elementId in self.theme_to_negative_element[new_name]:
                self.element_to_label[elementId] = list(map(lambda x: x.replace(theme, new_name), self.element_to_label[elementId]))

            # for elementId in self.elementto[new_name]:
            #     self.element_to_label[elementId] = list(map(lambda x: x.replace(theme, new_name), self.element_to_label[elementId]))
            self.stash_stuff()
            return{"status_code":200}
        except Exception as e:
            print("Exeption in rename theme")
            print(e)
            return{"status_code":500}

    def split_by_pattern(self, theme, patterns, new_theme_name):
        # create a new theme with the name new_theme_name
        self.add_theme(new_theme_name)

        #get results/explanation for original theme
        explanation = self.synthesizer_collector[theme].results["explanation"]

        pos_element_to_update =  dict()
        pos_element_to_update[new_theme_name] = []
        # print(explanation)

        #get examples from theme
        for elementId in self.theme_to_element[theme]:
        #get all sentences that match patterns in patterns and make a theme
            for pattern in patterns:
                matched_sentences = explanation[pattern]
                if(type(matched_sentences[elementId])==type(list())):
                    #remove this element from the old theme
                    self.delete_label(elementId, theme)

                    #add this element to the new theme
                    self.label_element(elementId, new_theme_name)

                    #add element to pos_elelemnt_to_update
                    pos_element_to_update[new_theme_name].append(elementId)

        res = dict()
        res['selected_theme'] = self.selected_theme
        res['all_themes'] = self.all_themes
        res["pos_update_labels"] = pos_element_to_update

        res["neg_update_labels"] = {}
        res['old_theme'] = theme
        res['new_themes'] = [new_theme_name]
        

        return res
    def split_theme(self, theme, group1, group2):
        group1_name = group1["name"]
        group2_name = group2["name"]


        if theme in self.all_themes:
            self.all_themes.remove(theme)
        pos_element_to_update = dict()
        pos_element_to_update[group1_name] = []
        pos_element_to_update[group2_name] = []

        neg_element_to_update = dict()
        neg_element_to_update[group1_name] = []
        neg_element_to_update[group2_name] = []


        #First group
        self.add_theme(group1_name)
        for elementId in group1["data"]:
            if(group1["data"][elementId][1]==1):
                self.label_element(elementId, group1_name)
                pos_element_to_update[group1_name].append(elementId)
            else:
                self.label_element(elementId, group1_name, positive=0)
                neg_element_to_update[group1_name].append(elementId)

        #Second group
        self.add_theme(group2_name)
        for elementId in group2["data"]:
            if(group2["data"][elementId][1]==1):
                self.label_element(elementId, group2_name)
                pos_element_to_update[group2_name].append(elementId)
            else:
                self.label_element(elementId, group2_name, positive=0)
                neg_element_to_update[group1_name].append(elementId)
        #return the new themes, selected_theme, user labels to update
        # self.selected_theme = group2_name
        res = dict()
        res['selected_theme'] = self.selected_theme
        res['all_themes'] = self.all_themes
        res["pos_update_labels"] = pos_element_to_update

        res["neg_update_labels"] = neg_element_to_update
        res['old_theme'] = theme
        res['new_themes'] = [group1_name, group2_name]

        self.stash_stuff()

        return res

    def get_user_labels(self, theme):
        if(theme not in self.theme_to_element):
            return {"status_code": 404, "message": "theme not found"}
        
        collector = {}
        for elementId in self.theme_to_element[theme]:
            collector[elementId] = [str(self.element_to_sentence[elementId]), 1]

        print("Labled data ", collector)
       
        
        return collector

    def bulk_label_element(self, ids, label, positive):
        for id in ids:
            self.label_element(id, label, positive)
        
        self.stash_stuff()
        return { "status":200, "message":"Sucess"}

    def label_element(self, elementId, label, positive=1):
        print("BINARY MODE ", elementId, label, positive)

        if(label not in self.all_themes):
            self.add_theme(label)
        if elementId in self.element_to_label:
            self.element_to_label[elementId].append(label)
        else:
            self.element_to_label[elementId] = [label]

        if elementId not in self.element_to_sentence:
            sentence = nlp(self.data[self.data["id"] == elementId]["example"].values[0])
            self.element_to_sentence[elementId] = sentence
            
            

        #handle negative annotation here
        if(self.binary_mode or True):
            
            #if its labeled positive check if it exists in negative collection and get rid of it
            if(positive==1 and label in self.theme_to_negative_element and elementId in self.theme_to_negative_element[label]):
                self.theme_to_negative_element[label].remove(elementId)
            #if its labeled negative check if it exists in positive collection. get rid of it and add it to negative collection
            elif(positive==0 ):
                if label in self.theme_to_element and elementId in self.theme_to_element[label]:
                    self.delete_label(elementId, label)
                
                #add elementId to negative example collection
                if label in self.theme_to_negative_element:
                    self.theme_to_negative_element[label].append(elementId)
                else:
                    self.theme_to_negative_element[label] = [elementId]
                
                print(self.theme_to_negative_element)
                self.stash_stuff()
                return {"status":200, "message":"negative label ok", "id":elementId, "label":label, "positive": positive}



        if label in self.theme_to_element:
            self.theme_to_element[label].append(elementId)
        else:
            self.theme_to_element[label] = [elementId]
        
        self.stash_stuff()
        return {"status":200, "message":"ok", "id":elementId, "label":label, "positive": positive}
    
    def delete_label(self, elementId, label):
        self.element_to_label[elementId].remove(label)

        self.theme_to_element[label].remove(elementId)

        print(self.theme_to_element)
        print(self.element_to_label)
        self.stash_stuff()
        return {"status":200, "message":"label deleted", "id":elementId, "label":label}

    def get_positive_and_negative_examples(self):
        try:
            positive_examples_id = self.theme_to_element[self.selected_theme]
        except:
            response = {}
            response["message"] = f"Nothing labeled for {self.selected_theme}"
            response["status_code"] = 404
            return response
        positive_examples = []
        for id in positive_examples_id:
            positive_examples.append(self.element_to_sentence[id])
        
        negative_examples_id = []
        negative_examples = []
        for elementId in self.element_to_label:
            if(not self.selected_theme in self.element_to_label[elementId]):
                negative_examples.append(self.element_to_sentence[elementId])
                negative_examples_id.append(elementId)
        print("negative data collection looks like ", self.theme_to_negative_element)
        #if binary mode is true add elements from negative collection
        if( self.selected_theme in self.theme_to_negative_element):
            for elementId in self.theme_to_negative_element[self.selected_theme]:
                if elementId not in negative_examples_id:
                    negative_examples.append(self.element_to_sentence[elementId])
                    negative_examples_id.append(elementId)
        

        return (positive_examples, negative_examples)

    def synthesize_patterns(self):
        
        #aggregate examples
        try:
            positive_examples_id = self.theme_to_element[self.selected_theme]
            if(len(positive_examples_id)==0):
                response = {}
                response["message"] = f"Nothing labeled for {self.selected_theme}"
                response["status_code"] = 300
                return response
        except:
            response = {}
            response["message"] = f"Nothing labeled for {self.selected_theme}"
            response["status_code"] = 300
            return response
        positive_examples = []
        for id in list(positive_examples_id):
            positive_examples.append(self.element_to_sentence[id])
        
        negative_examples_id = []
        negative_examples = []
        for elementId in list(self.element_to_label):
            if(not self.selected_theme in self.element_to_label[elementId]):
                negative_examples.append(self.element_to_sentence[elementId])
                negative_examples_id.append(elementId)
        print("negative data collection looks like ", self.theme_to_negative_element)
        #if binary mode is true add elements from negative collection
        if( self.selected_theme in self.theme_to_negative_element):
            for elementId in list(self.theme_to_negative_element[self.selected_theme]):
                if elementId not in negative_examples_id:
                    negative_examples.append(self.element_to_sentence[elementId])
                    negative_examples_id.append(elementId)

        #sanity check
        # print("POS ", positive_examples)
        # print("NEG ", negative_examples)
        

        #check if we have data annotated
        if len(positive_examples)==0:
            response = {}
            response["message"] = "Annotate Some More"
            response["status_code"] = 300
            return response
        
        #Check if data is in the chache
        cached = self.ran_cache()

        #For testing we will set the cache to none
        cached = None

        if(type(cached) != type(None)):
            df = cached
        else:
            self.synthesizer_collector[self.selected_theme].set_params(positive_examples+self.positive_phrases, negative_examples+self.negative_phrases)
            self.synthesizer_collector[self.selected_theme].find_patters()

            

            # df = self.save_cache(self.synthesizer_collector[self.selected_theme].patterns_set, positive_examples, negative_examples, positive_examples_id, negative_examples_id)
            try:
                df = self.save_cache(self.synthesizer_collector[self.selected_theme].patterns_set, positive_examples, negative_examples, positive_examples_id, negative_examples_id)
                self.synthesizer_collector[self.selected_theme].df_tracker = df
            except:
                response = {}
                response["message"] = "Annotate Some More"
                response["status_code"] = 300

                # print(self.synthesizer_collector[self.selected_theme].patterns_set)
                return response
        
        patterns = get_patterns(df, self.labels)


        return patterns
    
    def get_linear_model_results(self, refresh=False):
        #check if patterns are in cache, this will most likely be true because in most cases the above 
        #function will be called before this function
        #if patterns are not cached we will synthesize and cache them in this function

        if(refresh): #no need to resynthesize just pick new patters form the list
            df = self.synthesizer_collector[self.selected_theme].df_tracker
            res = train_linear_mode(df=df, data=self.data, 
            theme=self.selected_theme, soft_match_on=self.soft_match_on,words_dict=self.words_dict, 
            similarity_dict=self.similarity_dict, soft_threshold=self.soft_threshold, 
            pattern_customized_dict= self.pattern_customized_dict,
            deleted_patterns= self.synthesizer_collector[self.selected_theme].deleted_patterns,
            pinned_patterns= self.synthesizer_collector[self.selected_theme].pinned_patterns
            )
            return res


        #Check if data is in the chache    
        cached = self.synthesizer_collector[self.selected_theme].df_tracker

        #For testing 
        # cached = None
        if(type(cached) != type(None)): #for test
            df = cached
        else:
            res = self.synthesize_patterns()
            
            
            if("status_code" in res and( res["status_code"]==404 or res['status_code']==300)):
                return res
            
            cached = self.ran_cache() 
            #TODO annotate some more data needs to be added here and maybe a way to check if we have enough data annotated
            df = cached
            if(type(df) == type(None)):
                #TODO handle this in a better way, but if the code gets here we know that no patterns have been synthesized becuase there weren't enough annotations
                return {"message": f"Not enough annotations for {self.selected_theme}"}
        res = train_linear_mode(df=df, data=self.data, theme=self.selected_theme, soft_match_on=self.soft_match_on,words_dict=self.words_dict, similarity_dict=self.similarity_dict, soft_threshold=self.soft_threshold, 
        pattern_customized_dict=self.pattern_customized_dict, deleted_patterns= self.synthesizer_collector[self.selected_theme].deleted_patterns,
            pinned_patterns= self.synthesizer_collector[self.selected_theme].pinned_patterns)

        return res

    def delete_softmatch(self, pattern, soft_match):
        #TODO delete from similarity list

        if not pattern in self.pattern_customized_dict:
            self.pattern_customized_dict[pattern] = {}
        self.pattern_customized_dict[pattern][soft_match] = -1
        if pattern.endswith('+*'):
            if not pattern[:-2] in self.pattern_customized_dict:
                self.pattern_customized_dict[pattern[:-2]] =  {}
            self.pattern_customized_dict[pattern[:-2]][soft_match] = -1

        response = {}
        response["message"] = f"Deleted {soft_match} from {pattern}'s list"
        response["status_code"] = 200


        return response

    def add_softmatch(self, pattern, soft_match):
        #TODO add to similarity list
        
        if not pattern in self.pattern_customized_dict:
            self.pattern_customized_dict[pattern] = {}
        self.pattern_customized_dict[pattern][soft_match] = 1
        if pattern.endswith('+*'):
            if not pattern[:-2] in self.pattern_customized_dict:
                self.pattern_customized_dict[pattern[:-2]] =  {}
            self.pattern_customized_dict[pattern[:-2]][soft_match] = 1

        response = {}
        response["message"] = f"Deleted {soft_match} from {pattern}'s list"
        response["status_code"] = 200


        return response

    def delete_softmatch_globally(self, pivot_word_remote, similar_word):
        #TODO delete from pivot_word's list
        
        pivot_word = pivot_word_remote[1:-1]
        print("deleting globally ", pivot_word, similar_word)
        if pivot_word in self.similarity_dict and similar_word in self.similarity_dict[pivot_word]:
            print("Before ", self.similarity_dict[pivot_word])

            del self.similarity_dict[pivot_word][similar_word]

            print("After", self.similarity_dict[pivot_word])

        response = {}
        response["message"] = f"Deleted {similar_word} from {pivot_word}'s list"
        response["status_code"] = 200


        return response

    def add_softmatch_globally(self, pivot_word, similar_word):
        #TODO add to pivot_word's list
        
        if pivot_word in self.similarity_dict:
            self.similarity_dict[pivot_word][similar_word] = 1.0

        response = {}
        response["message"] = f"Added {similar_word} from {pivot_word}'s list"
        response["status_code"] = 200


        return response

    def run_multi_label_test(self, iteration, no_annotation):
        #I declare thee a results collector
        collector = []

        #keep track of how many are being annotated
        total_annotation_count = 0
        theme_annotation_count = {}
        all_themes = self.get_themes()

        for theme in all_themes:
            theme_annotation_count[theme] = 0


        #get all ids
        all_ids  = self.data["id"].values.tolist()


        for j in range(iteration):
            #pick random ids to annotate
            ids = random.sample(all_ids, no_annotation)

            #for each id picked annotate all the positive labels
            for i in range(len(ids)):
                total_annotation_count += 1
                for theme in all_themes:
                    if(self.data[self.data['id']== ids[i]][theme].values[0]):
                        self.label_element(ids[i], theme)
                        theme_annotation_count[theme] += 1 
                        print(f"labeled example {ids[i]} as {theme}")

            #synthesize and come up with scores for each theme
            all_themes_results = []
            for theme in all_themes:
                print(f"working with {theme}")
                self.set_theme(theme)
                temp = {}
                try:
                    results = self.get_linear_model_results()
                except Exception as Argument:
                    # creating/opening a file
                    f = open("errorlog.txt", "a")
                
                    # writing in the file
                    f.write(str(Argument))
                    
                    # closing the file
                    f.close()  

                
                #collect only relevant information from the results
                try:
                    temp["theme"] = theme
                    temp["fscore"] = results["fscore"]
                    temp["recall"] = results["recall"]
                    temp["precision"] = results["precision"]
                    temp["overall_fscore"] = results["overall_fscore"]
                    temp["overall_recall"] = results["overall_recall"]
                    temp["overall_precision"] = results["overall_precision"]
                    temp["patterns"] = results["patterns"]
                    temp["weights"] = results["weights"]
                    temp["total_annotation_count"] = total_annotation_count
                    temp["annotation_per_theme"] = theme_annotation_count
                except:
                    temp["theme"] = theme
                    temp["message"] = results
                


                all_themes_results.append(temp)
            macro_counter = 0
            macro_score_fpr = [0, 0, 0]
            macro_overall_score_fpr = [0, 0, 0] 
            
            for theme_result in all_themes_results:
                #collect values for macro scores
                if('fscore' in theme_result):
                    macro_counter += 1
                    macro_score_fpr[0] += theme_result['fscore']
                    macro_score_fpr[1] += theme_result['precision']
                    macro_score_fpr[2] += theme_result['recall']
                    
                    macro_overall_score_fpr[0] += theme_result['overall_fscore']
                    macro_overall_score_fpr[1] += theme_result['overall_precision']
                    macro_overall_score_fpr[2] += theme_result['overall_recall']
            if(macro_counter==0): #no patterns were synthesized
                continue

            temp = {}

            temp['macro_scores_fpr'] = [x/macro_counter for x in macro_score_fpr]
            temp['macro_overall_scores_fpr'] = [x/macro_counter for x in macro_overall_score_fpr]
            temp['annotation_count'] = total_annotation_count

            all_themes_results.append(temp)
            

            collector.append(all_themes_results)


        #save results to file
        with open('results/multilabel_aggregate_results.json', 'w') as f:
            json.dump(collector, f)





        #return resuts
        return collector

    def toggle_binary_mode(self, binary_mode):
        self.binary_mode = bool(binary_mode)
        print("Binary mode = ", self.binary_mode)
        
        response = {}
        response["message"] = f"binary_mode: {binary_mode}"
        response["status_code"] = 200

        self.stash_stuff()
        return response
    
    
    def get_bert_annotation(self):
        positive_examples, negative_examples = self.get_positive_and_negative_examples()

        try:
            ids = self.data["id"].values.tolist()
            annotations ={}
            res = NN_classificaion(positive_examples, negative_examples, self.data['example'].values, epoch_num=5)
            print(res)
            for i in range(len(res)):
                annotations[ids[i]] = res[i]
            
            return annotations
        except Exception as e:
            print(e)
            response = {}
            response["message"] = f"Something went wrong"
            response["status_code"] = 500
            
            return response

    def get_NN_cluster(self):
        try:
            res = {}
            res['data'] = NN_cluster(self.data)

            
            res['group_names'] = [f"Group-{x+1}" for x in range(len(res['data']))]
            return res
        except:
            response = {}
            response["message"] = f"Something went wrong"
            response["status_code"] = 500


            return response

    def get_NN_classification(self):
        ids = self.data["id"].values.tolist()
        try:
            ids = self.data["id"].values.tolist()
            groups ={}
            res =NN_multi_classificaion(self.theme_to_element, self.theme_to_negative_element, self.all_themes, {k:str(v) for k,v in self.element_to_sentence.items()}, examples=self.data["example"].values)
            for i in range(len(res)):
                if(res[i] in groups):
                    groups[res[i]].append(ids[i])
                else:
                    groups[res[i]] = [ids[i]]
            response = {}
            response['data'] = list(groups.values())
            response['group_names'] = [f"Predicted-{self.all_themes[x]}" for x in range(len(response['data']))]
            return response
        except Exception as e:
            print(e)
            response = {}
            response["message"] = f"Something went wrong"
            response["status_code"] = 500
            
            return response
    
    def get_original_dataset_order(self):
        res = {}
        res['data'] = [self.data["id"].values.tolist()]
        res['group_names'] = [f"Group - {x+1}" for x in range(len(res['data']))] 
        return res

    def get_pattern_clusters(self):
        if(self.selected_theme not in self.all_themes):
            response = {}
            response["message"] = f"We need more annotated data to do this"
            response["status_code"] = 300
            return response
        if("patterns" not in self.synthesizer_collector[self.selected_theme].results or len(self.synthesizer_collector[self.selected_theme].results['patterns'])==0):
            
            response = {}
            response["message"] = f"We need more annotated data to do this"
            response["status_code"] = 300
            
            return response

        patterns = self.synthesizer_collector[self.selected_theme].results['patterns']

        explanation = self.synthesizer_collector[self.selected_theme].results['explanation']


        # ids = self.data["id"].values.tolist()
        # res = pattern_clusters(patterns, explanation, ids)
        # print(res)
        try:
            res = {}
            ids = self.data["id"].values.tolist()
            res['data'] = pattern_clusters(patterns, explanation, ids)
            res["group_names"] = [x['pattern'] for x in patterns]
            res["group_names"].append("Unmatched")

            return res
        except:
            response = {}
            response["message"] = f"Something went wrong"
            response["status_code"] = 500
            
            return response
        
    def pin_pattern(self, theme, pattern):
        self.synthesizer_collector[theme].pinned_patterns.append(pattern)
        if(pattern in self.synthesizer_collector[theme].deleted_patterns ):
            self.synthesizer_collector[theme].deleted_patterns.remove(pattern)
        self.stash_stuff()
        return {"status_code":200, "message":f"Pattern {pattern} pinned in {theme}"}

    def delete_pattern(self, theme, pattern):
        print(f"Deleting pattern {pattern} from {theme}")
        self.synthesizer_collector[theme].deleted_patterns.append(pattern)

        if(pattern in self.synthesizer_collector[theme].pinned_patterns ):
            self.synthesizer_collector[theme].pinned_patterns.remove(pattern)
        
        self.stash_stuff()
        return {"status_code":200, "message":f"Pattern {pattern} deleted from {theme}"}