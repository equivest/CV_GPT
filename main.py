
job_ad_start = """About the job
Would you like to work at the intersection between data science, machine learning and sales? Then we look forward to hearing from you!

Data is at the centre of reimaging business, enabling our clients to do things differently and to capture new opportunities. As the new Data Science Sales Lead, you will be leading the Data Science and Machine Learning Capabilities within Accenture Applied Intelligence in Norway, focusing on complex and innovative solutions to meet client’s digitalization goals.

How do you know this could be the role for you? 

Lead a strong and highly skilled AI/ML/DS capability which consists of 25 people

Build strong relationships with key stakeholders, i.e., Client Account Leads, as well as colleagues both on a national and international level

Drive sales processes and expand current DS/AI/ML footprint at our clients

Ensure that relevant digital strategies, policies, standards and practices are applied correctly across technology programs/projects

Qualifications: 

University degree in Computer Science, Data Science, Mathematics of similar field

Minimum of 10 years of experience within the IT industry, particularly in sales

Excellent skills in data science, AI and ML

Good understanding of Cloud technologies (GCP/ Azure)

Strong organizational and leadership skills

Experience of working in an agile environment

What´s In It For You

Competitive salary, good bonus schemes, occupational pension schemes, share savings and that parental leave is covered beyond NAV's conditions up to 12G

Professional development including your own personal budget for skills development. This can be used for courses, certification, conferences, or other learning activities

Monthly community meetings with a focus on personal and professional development

Larger and smaller social gatherings such as payday parties and annual parties

A wide range of sports and interest groups, including football, running, mountain sports and e-sports

The position will report to Head of Accenture Applied Intelligence Norway and workplace is Oslo, Norway. Interested?

Apply online by attaching your CV, application letter and educational diplomas as soon as possible. We look forward to receiving your application.

Please note that your application will not be reviewed during summer vacation from 1st July to 31 July 2023. This will be done upon our return in the first week of August 2023.

If you have any questions concerning the position, please reach out to:

Johan Ekström - phone: +47 94169655, Managing Director and Head of Accenture Applied Intelligence Norway. If you have questions concerning the application and/or recruitment process, please contact Raluca Rohatin - phone + 47 900 72 459, Recruiting Specialist – Strategy & Consulting.

About The Department / Team

Applied Intelligence is how Accenture uses Artificial Intelligence (AI), automation, and analytics to reimagine business—enabling our clients to do things differently and do different things. Our unique approach breaks down silos and creates more agile and adaptive processes, enabling better decision making and empowering businesses to identify and capture completely new opportunities.

The Applied Intelligence team in Norway consists of 40 professionals with diverse experience and backgrounds including statistics, engineering, economics, and other quantitative disciplines.

About Us

Accenture is a leading global professional services company, providing a broad range of services and solutions in strategy, consulting, digital, technology and operations. With more than 710,000 people serving clients in more than 120 countries, Accenture drives innovation to improve the way the world works and lives.
"""


cv_start = """Oscar Fictitious Almquist

Oslo, Norway | +47 50125849
oscaralmquist@datalead.no

I bring 12+ years of rich experience to the table, and specialize in steering data-driven sales initiatives with a proven track record of capturing new market opportunities through innovative AI and ML solutions. With my leadership, I've managed to translate complex data solutions into tangible business benefits, making me adept at nurturing client relationships, expanding the AI/ML footprint, and ensuring digital compliance at every touchpoint.

EXPERIENCE

Head of Data Sales
SynthTech Solutions, Oslo
June 2015 - Present

Oversaw a dedicated team of 20 data scientists and ML experts.
Strategized and delivered growth, resulting in a 35% increase in annual sales.
Pioneered AI-based sales solutions, generating over $10M in new business opportunities.
Collaborated with international branches to devise global data strategies.
Senior Data Science Sales Manager
EvoTech, Oslo
Jan 2012 - May 2015

Managed major client portfolios, driving AI and ML sales upwards by 50%.
Introduced agile practices, optimizing sales processes and client deliverables.
Conducted workshops, bridging the gap between technology and sales teams.
Data Solutions Sales Consultant
NeuroFlow Technologies, Oslo
June 2008 - Dec 2011

Spearheaded Cloud technology (GCP/Azure) sales, with a focus on data services.
Achieved a consistent sales growth rate of 20% year-over-year.
Liaised with IT departments, ensuring seamless delivery and post-sales support.
EDUCATION

Master's in Data Science
University of Oslo
Aug 2006 - Jun 2008

Bachelor's in Computer Science
Norwegian University of Science and Technology, Trondheim
Aug 2003 - Jun 2006

SKILLS

Expertise in AI, ML, and Data Science
Proficient in Cloud Technologies (GCP/Azure)
Strong Sales Acumen
Agile Methodologies
Leadership & Team Management
Stakeholder Communication & Relationship Building
CERTIFICATIONS

Certified Sales Specialist in AI & ML - 2020
Advanced Cloud Technology Professional (GCP/Azure) - 2019
Agile Leadership - 2017
ASSOCIATIONS

Member, Oslo Data Science Association
Active Participant, AI Norway Meetups
Please reach out to me directly or through my LinkedIn profile. I am eager to bring my expertise to Accenture Applied Intelligence and believe that together, we can further amplify the transformative power of AI for businesses in Norway.

[LinkedIn Profile Link]

Note: Kindly be informed that I will be on vacation from 1st July to 31 July 2023 and will be available for discussions post that."""


















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

# Initial values for demo

# Persistent state across reruns
if 'scoring_system' not in st.session_state:
    st.session_state.scoring_system = ""

# Step 1: User provides job advertisement text
st.subheader("Step 1: Enter Job Ad Text")
job_ad = st.text_area("Job Ad:", value=job_ad_start if 'job_ad_value' not in st.session_state else st.session_state.job_ad_value)

if st.button("Generate Scoring System"):
    st.session_state.scoring_system = generate_scoring_system(job_ad)
    corrected_table = fix_table_format(st.session_state.scoring_system)
    st.markdown(f"**Scoring System:**\n\n{corrected_table}")
    st.session_state.job_ad_value = job_ad

# If we have a scoring system, we allow for CV input and scoring
if st.session_state.scoring_system:
    # Step 2: User provides CV text
    st.subheader("Step 2: Enter CV Text")
    cv_text = st.text_area("CV:", value=cv_start if 'cv_text_value' not in st.session_state else st.session_state.cv_text_value, key='cv_text_area')

    if st.button("Score CV", key='score_cv_button'):
        cv_score = score_cv(cv_text, st.session_state.scoring_system)
        corrected_table = fix_table_format(cv_score)
        st.markdown(f"**CV Score:**\n\n{corrected_table}")
        st.session_state.cv_text_value = cv_text


