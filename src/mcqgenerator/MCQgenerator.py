import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# loading json file 
with open("C:/Users/Suraj/Desktop/Python/mcqgenerator/response.json", "r") as f:
    RESPONSE_JSON = json.load(f)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=api_key,
    model="mixtral-8x7b-32768",
    temperature=0.5
)

template = """
Text: {text}
You are an expert MCQ maker.Given the above text, it is your job to create a quiz of {number} multiple choice question for {subject} student in {tone} tone. Make Sure the questions are not repeated and check all the questions for {subject} student in {tone} tone. Make sure the question are not repeated and check all the questions to be confoming the text as well. Make sure to format your response like response_json below and use it as a guide. Ensure to make {number} MCQs.

## response_json
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template,
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_template = PromptTemplate(
    input_variables=["quiz", "subject"],
    template=template2
)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_template, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)