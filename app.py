import torch
import json
import streamlit as st

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
except:
    print("Import Error")

from newsParser import getNewsTitleText


def t5TextSummerizer(text):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    # print("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    summary_ids = model.generate(tokenized_text,
                                 num_beams=14,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=100,
                                 early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Summarized text: \n", output)

    return output


st.title("Text Summarization")

option = st.selectbox("Select the option for Summarization: ",
                      ["Text", 'URL'])

# print the selected topic
st.write("You have selected: ", option)
if option == "Text":
    text = st.text_area(label="Input Text", value="", height=200)
    if st.button("Generate") and text != "":
        summary = t5TextSummerizer(text)
        st.text_area(label="Summary", value=summary, height=200)
elif option == "URL":
    url = st.text_input("Input URL", value="")
    print("url =", url)
    if st.button("Generate") and url != "":
        title, text = getNewsTitleText(url)
        summary = t5TextSummerizer(text)
        st.text_area(label="Summary", value=summary, height=200)

url = "http://timesofindia.indiatimes.com/world/china/chinese-expert-warns-of-troops-entering-kashmir/articleshow/59516912.cms"

# st.text_area(label=topic, value=response, height=700)
