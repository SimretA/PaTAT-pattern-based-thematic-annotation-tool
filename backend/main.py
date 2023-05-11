from fastapi import FastAPI, Cookie
from synthesizer.api_helper import *
from api.schemas.Theme import *
from api.schemas.Labeling import *
from synthesizer.penality_based_threaded import Synthesizer 
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
import asyncio
import time
from pydantic import BaseModel
from typing import List
from fastapi import Request

import pandas as pd

import random
import torch
import numpy as np

torch.multiprocessing.set_start_method('spawn', force=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

executor = ProcessPoolExecutor()
loop = asyncio.get_event_loop()


app = FastAPI()

user_to_apiHelper = {}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class LablingModel(BaseModel):
    theme: str
    elementId: str = None
    phrase: str = None
    positive: int = 1
    pattern: str = None



class Item(BaseModel):
    depth: int
    rewardThreshold: float
    penalityThreshold: float
    featureSelector: int
class BulkLabel(BaseModel):
    ids: List[str]
    label: str
    positive: str
class PatternsSplitThemeItem(BaseModel):
    patterns: List[str]
    new_theme_name: str
    theme: str

@app.get("/")
async def home():
    return {"status":"Running"}


@app.get("/bert_annotation")
async def bert_annotation(request:Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].get_bert_annotation)
    return results

@app.get("/restore_session/{user}")
async def restore_session(user: str):
    return {}


@app.get("/create_session/{user}")
async def create_session(user: str):
    print("The user in create session is ", user)
    if(user not in user_to_apiHelper):
        user_to_apiHelper[user] = APIHelper(user=user)
    return "Done"

threadpool = {}

###v1 endpoints
@app.get("/dataset")
async def get_labeled_examples(request: Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    
    if(user not in user_to_apiHelper):
        user_to_apiHelper[user] = APIHelper(user=user)
    # print("coookiieee", user)
    return user_to_apiHelper[user].get_labeled_dataset()



@app.post("/phrase")
async def label_element_by_phrase(request: Request, body: LablingModel):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = user_to_apiHelper[user].label_by_phrase(body.phrase, body.theme, body.positive, body.elementId)
    return results

@app.post("/clear")
async def clear_labeling(request:Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    return user_to_apiHelper[user].clear_label()

@app.get("/combinedpatterns")
async def combinedpatterns(request:Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].resyntesize)
    user_to_apiHelper[user].results = results
    return results

@app.get("/test/{iteration}/{annotation}")
async def test(iteration:int, annotation: int, body:Item):
    print(body)
    start =  time.time()
    results = user_to_apiHelper['simret'].run_test(iteration, annotation, depth= body.depth, rewardThreshold=body.rewardThreshold, penalityThreshold=body.penalityThreshold)
    end = time.time()
    print(results)
    results[0]['time'] = end-start

    return results

@app.get("/themes")
async def get_themes(request:Request):
    user = request.headers.get("annotuser")
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    return user_to_apiHelper[user].all_themes

@app.post("/add_theme")
async def add_theme(request:Request, body:ThemeName):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    return user_to_apiHelper[user].add_theme(body.theme)


@app.post("/delete_theme")
async def delete_theme(request:Request, body:ThemeName):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    return user_to_apiHelper[user].delete_theme(body.theme)

@app.get("/selected_theme")
async def get_selected_theme(request:Request):
    user = request.headers.get("annotuser")
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    if(user_to_apiHelper[user].get_selected_theme() in user_to_apiHelper[user].all_themes):
        return user_to_apiHelper[user].get_selected_theme()
    else:
        return None


@app.post("/set_theme")
async def set_theme(request: Request, body:ThemeName):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    return (body.theme,user_to_apiHelper[user].set_theme(body.theme))

@app.get("/related_examples/{id}")
async def get_related_examples(request:Request, id:str):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].get_related, id)
    return results

@app.get("/explain/{pattern}")
async def explain_pattern(request: Request, pattern:str):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].explain_pattern, pattern)
    return results

def main():
    synthh = Synthesizer(positive_examples = "examples/price_big", negative_examples = "examples/not_price_big")
    print(synthh.find_patters(outfile="small_thresh"))
# main() pid 31616

######################################################################################################################
@app.post("/delete_label")
async def delete_label(request:Request, body: LablingModel):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].delete_label, body.elementId, body.theme)
    res = await future1
    return res


@app.post("/merge_themes")
async def merge_themes(request:Request, body:MergeThemeItem):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].merge_themes, body.theme1, body.theme2, body.new_theme)
    res = await future1
    return res


@app.post("/split_theme")
async def split_themes(request:Request, body: SplitThemeItem):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].split_theme, body.theme, body.group1, body.group2)
    res = await future1
    return res

@app.post("/split_theme_by_pattern")
async def split_themes_by_pattern( request:Request, body: PatternsSplitThemeItem):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].get_examples_by_patterns, body.theme, body.patterns, body.new_theme_name)
    res = await future1
    return res

@app.post("/rename_theme")
async def split_themes_by_pattern(request:Request, body: RenameTheme):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].rename_theme, body.theme, body.new_name)
    res = await future1
    return res


@app.post("/label")
async def label_example(request:Request, body: LablingModel):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].label_element, body.elementId, body.theme, body.positive)
    res = await future1
    return res

@app.post("/bulk_label")
async def bulk_label_example(request:Request, body: BulkLabel):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    future1 = loop.run_in_executor(None, user_to_apiHelper[user].bulk_label_element, body.ids, body.label, body.positive)
    res = await future1
    return res

@app.post("/labeled_data")
async def labeled_data(request:Request, body: ThemeName):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].get_user_labels, body.theme)
    return results



@app.get("/patterns")
async def patterns(request: Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }

    results = await loop.run_in_executor(executor, user_to_apiHelper[user].synthesize_patterns)
    if("status_code" in results and( results["status_code"]==404 or results['status_code']==300)):
        return results
    user_to_apiHelper[user].synthesizer_collector[user_to_apiHelper[user].selected_theme].patterns = results
    print(results)
    return results

@app.get("/annotations")
async def annotations(request:Request, refresh:bool = False):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].get_linear_model_results, refresh)
    user_to_apiHelper[user].synthesizer_collector[user_to_apiHelper[user].selected_theme].results = results
    
    return results

@app.get("/delete_softmatch/{pattern}/{softmatch}")
async def delete_softmatch(request:Request, pattern:str, softmatch:str):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = await loop.run_in_executor(executor, user_to_apiHelper[user].delete_softmatch, pattern, softmatch)
    return results

@app.get("/delete_softmatch_globally/{pivot_word}/{similar_word}")
async def delete_softmatch_globally_end(request:Request, pivot_word:str, similar_word:str):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results = user_to_apiHelper[user].delete_softmatch_globally(pivot_word, similar_word)
    return results

@app.get("/toggle_binary_mode/{binary_mode}")
async def toggle_binary_mode(request:Request, binary_mode:int):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results =  user_to_apiHelper[user].toggle_binary_mode(binary_mode)
    return results



@app.post("/delete_pattern")
async def delete_pattern(request:Request, body: LablingModel):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    res = user_to_apiHelper[user].delete_pattern(body.theme, body.pattern)
    return res


@app.post("/pin_pattern")
async def pin_pattern(request:Request, body: LablingModel):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    res = user_to_apiHelper[user].pin_pattern(body.theme, body.pattern)
    return res

@app.get("/NN_cluster")
async def NN_cluster(request: Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results =  user_to_apiHelper[user].get_NN_cluster()
    return results

@app.get("/NN_classification")
async def NN_classification(request:Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results =  user_to_apiHelper[user].get_NN_classification()
    return results

@app.get("/original_dataset_order")
async def original_dataset_order(request: Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results =  user_to_apiHelper[user].get_original_dataset_order()
    return results

@app.get("/pattern_clusters")
async def pattern_clusters(request: Request):
    user = request.headers.get('annotuser')
    if(user=="null" or user==None):
        return{
            "status_code":404,
            "message": "Unauthorized"
        }
    results =  user_to_apiHelper[user].get_pattern_clusters()
    return results

@app.get("/test_multilabel/{iteration}/{no_annotation}")
async def test_multilabel(iteration:int, no_annotation:int):
    results = await loop.run_in_executor(executor, user_to_apiHelper['simret'].run_multi_label_test, iteration, no_annotation)
    return results

