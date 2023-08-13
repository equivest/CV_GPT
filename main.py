import streamlit as st
#import json
import pandas as pd
import numpy as np
import os

#load environment variables
import ast
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
G_KEY = os.environ["G_KEY"]
G_SCOPES = ast.literal_eval(os.environ["G_SCOPES"])
G_PRJ_ID = os.environ["G_PRJ_ID"]


from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
chat4 = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, model="gpt-4")
chat3 = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
#4096 tokens = 16384, BUT crashes on 12000 characters. That is 0.7 of the total tokens
chat3_long = ChatOpenAI(temperature=.7, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-16k")



build_scoring = """Act as a recruiter for data professionals such as data scientists, data engineers and data analysts_

I want you to extract the requirements for this role and build a table for scoring candidates. The table should have the following columns:
Description if the requirement
Candidate fitness score (0-5)
Description of how to score the fitness 
Weight of importance (0-1, based on your understanding of the job ad)
Weighted score (fitness score * weight)

Return a table and last row for Total Score

Here is the job ad:"""


do_scoring = """Use the scoring system you made to score this candidate CV:"""

do_json = """Return the scoring result table as json only. No text before or after"""


def generate_scoring_system(job_ad):
    result = chat4(
        [
            #SystemMessage(content="You are a stock analyst with expertise in mineral exploration. Clean the text and slightly summarize if too long. Skip disclaimers about forward looking statements etc. Here is the document: "),
            SystemMessage(content=build_scoring),
            HumanMessage(content=job_ad) # 1 token is 4 characters, but we multiply by 2.2 because we also need to count the output
        ]
    )
    return result.content

#scoring_system = generate_scoring_system(job_ad)

def score_cv(cv,scoring_system):
    result = chat4(
        [
            #SystemMessage(content="You are a stock analyst with expertise in mineral exploration. Clean the text and slightly summarize if too long. Skip disclaimers about forward looking statements etc. Here is the document: "),
            SystemMessage(content=build_scoring),
            HumanMessage(content=job_ad), # 1 token is 4 characters, but we multiply by 2.2 because we also need to count the output
            AIMessage(content=scoring_system),
            HumanMessage(content=do_scoring+cv)
        ]
    )
    return result.content

#cv_score = score_cv(cv,scoring_system)

def return_json(cv,scoring_system):
    result = chat4(
        [
            #SystemMessage(content="You are a stock analyst with expertise in mineral exploration. Clean the text and slightly summarize if too long. Skip disclaimers about forward looking statements etc. Here is the document: "),
            SystemMessage(content=build_scoring),
            HumanMessage(content=job_ad), # 1 token is 4 characters, but we multiply by 2.2 because we also need to count the output
            AIMessage(content=scoring_system),
            HumanMessage(content=do_scoring+cv)
        ]
    )
    return result.content


def return_json(cv_score):
    result = chat4(
        [
            #SystemMessage(content="You are a stock analyst with expertise in mineral exploration. Clean the text and slightly summarize if too long. Skip disclaimers about forward looking statements etc. Here is the document: "),
            SystemMessage(content=do_json),
            HumanMessage(content=cv_score)#, # 1 token is 4 characters, but we multiply by 2.2 because we also need to count the output
            #AIMessage(content=scoring_system),
            #HumanMessage(content=do_scoring+cv)
        ]
    )
    return result.content

#def fix_table_format(table_str):
#    lines = table_str.split("\n")
#    # Add the missing separator after the header
#   if len(lines) > 2 and "|---" in lines[1]:
#        lines.insert(2, "|---|---|---|---|---|")
#    return "\n".join(lines)

def fix_table_format(table_string):
    """
    Fixes markdown table formatting.
    """
    lines = table_string.split('\n')
    
    # Replacing the problematic line with the correct format
    for idx, line in enumerate(lines):
        if line.startswith('|---'):
            lines[idx] = '|---|---|---|---|---|'  # Assuming table always has 5 columns

    return '\n'.join(lines)



st.title("Job Ad and CV Scoring")

# If these keys don't exist in the session state, create them
if 'job_ad' not in st.session_state:
    st.session_state.job_ad = ""
if 'cv_text' not in st.session_state:
    st.session_state.cv_text = ""
if 'scoring_system' not in st.session_state:
    st.session_state.scoring_system = ""
if 'cv_score' not in st.session_state:
    st.session_state.cv_score = ""

# Step 1: User provides job advertisement text
st.subheader("Step 1: Enter Job Ad Text")
st.session_state.job_ad = st.text_area("Job Ad:", value=st.session_state.job_ad)

if st.button("Generate Scoring System"):
    st.session_state.scoring_system = generate_scoring_system(st.session_state.job_ad)

# Display the scoring system if it exists
if st.session_state.scoring_system:
    st.markdown(fix_table_format("Scoring System:\n" + st.session_state.scoring_system), unsafe_allow_html=True)

    # Step 2: User provides CV text
    st.subheader("Step 2: Enter CV Text")
    st.session_state.cv_text = st.text_area("CV:", value=st.session_state.cv_text, key='cv_text_area')

    if st.button("Score CV", key='score_cv_button'):
        st.session_state.cv_score = score_cv(st.session_state.cv_text, st.session_state.scoring_system)

# Display the CV score if it exists
if st.session_state.cv_score:
    st.markdown(fix_table_format("CV Score:\n" + st.session_state.cv_score), unsafe_allow_html=True)


