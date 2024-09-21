import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.MCQgenerator import generate_evaluate_chain

load_dotenv()


# loading json file 
with open("C:/Users/Suraj/Desktop/Python/mcqgenerator/response.json", "r") as f:
    RESPONSE_JSON = json.load(f)

# creating title for app
st.title("MCQ Generator App")

# create a form using st.form
with st.form("user_inputs"):
    # file upload
    uploaded_file = st.file_uploader("Upload a pdf or text file")

    # input fields
    mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50)

    # Subject
    subject = st.text_input("Insert subject", max_chars=20)

    # Quiz tone
    tone = st.text_input("Complexity level of questions", max_chars=20, placeholder="Simple")

    # Submit button
    button = st.form_submit_button("Generate MCQs")

    # check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON),
                        }
                    )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error(e)

            else:
                print(f"Total tokens : {cb.total_tokens}")
                print(f"Total Cost : {cb.total_cost}")
                if isinstance(response, dict):
                    # Extract the quiz data from response
                    quiz = response.get("quiz")
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.dataframe(df)
                            # Display the review in a text box as well
                            st.text_area(label="Review", value=response.get("review"))
                        else:
                            st.error("Error in table data")
                else:
                    st.write(response)
                
