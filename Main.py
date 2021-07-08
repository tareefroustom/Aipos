import streamlit as st

#The Named Entity Recognition highligting library
from spacy import displacy


st.set_page_config(page_title="Aipos", page_icon="💻")#, layout="wide")


def Censor(Token, TokenText):
    #Step one
    Pass = True
    if Token[1][2:] in ["rd","str","rem"]:
        if len(TokenText.split(" ")) < 2:
            Pass = False
    if Token[1][2:] in ["pop"]:
        if len(TokenText.split(" ")) < 2:
            if TokenText.split(" ") == "patient":
                Pass = False

    if len(TokenText) == 2:
        Pass = False

    if Token[1][2:] in ["product"]:
        if TokenText[:-1] == "the":
            Pass = False

    return Pass

def UntangleTokens(TokenText, Tokens):
    Pass = True
    for Token in Tokens:
        if TokenText in " ".join(Token[0]):
            Pass = False

    return Pass


if __name__ == '__main__':

    #load the library
    import ktrain

    # load the model
    NerPredictor = ktrain.load_predictor("./July-07-2021-ca_8500_Sentences_News_Info_Extraction_1")#July-06-2021-ca_8500_Sentences_News_Info_Extraction_5

    st.markdown( """<head><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Titillium Web"><style> body { font-family: "Titillium Web", sans-serif;} </style></head>""",
                unsafe_allow_html=True)
    st.sidebar.write("""<div style="font-size: 22px; font-family:Titillium Web ; color:#f1f6f9">Aipos: Market insights at Scale </div>""",
                unsafe_allow_html=True)


    # Set the header image
    page_bg_img = '''<style>
                            .css-1v3fvcr {
                                    background-image: url("https://i.imgur.com/pEryXvM.png");
                                    background-repeat: no-repeat;
                                    background-size: auto;
                                  }
                    </style>'''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.text("")
    st.text("")


    # Change the sidebar background color
    st.markdown("""<style> .css-1l02zno { font-family:Titillium Web; background-color: #151D48; color:#ffffff } </style> """,
                unsafe_allow_html=True)


    Link = st.text_input('Enter a link', "")

    if Link:

        import re
        import nltk
        import nltk.data

        from goose3 import Goose

        Goo = Goose()
        Article = Goo.extract(url=Link)
        nltk.download('punkt')
        Tokenizer  = nltk.data.load('tokenizers/punkt/english.pickle')


        st.write(f"""<div style="font-size: 18px; font-family:Titillium Web">Articlle's title: {Article.title}</div>""",
                         unsafe_allow_html=True)

        st.text("")
        st.text("")

        for Sen in Tokenizer.tokenize(Article.cleaned_text):

            Classes = []

            st.write(
                f"""<div style="font-size: 15px; font-family:Titillium Web">{Sen}</div>""",
                unsafe_allow_html=True)
            st.text("")

            Sen = re.sub("(?<=\d)\.(?=\d)", ",", Sen)
            Sen = re.sub("—", ",",               Sen)
            Sen = re.sub(":", ",",               Sen)

            PredsUnfiltered = NerPredictor.predict(Sen.lower(), return_proba=True)
            Preds           = []
            for Pred in PredsUnfiltered:
                if Pred[1][0] == "b" and Pred[2] < 0.3:
                    Preds.append((Pred[0], "i-" + Pred[1][2:], Pred[2]))
                else:
                    Preds.append(Pred)


            Tokens = []
            Tags   = []

            st.text(Preds)

            for n, i in enumerate(Preds):

                inew = []

                if i[1][0] == "b" and i[1][2:] not in ["product", "area", "con", "pop"]:

                    inew.append(i[0])

                    for n1, i1 in enumerate(Preds[n + 1:]):
                        if   i1[1][0] != "b":
                             inew.append(i1[0])
                        elif i1[1][0] == "b" and i1[1][2:] == i[1][2:]:
                             inew.append(i1[0])
                        else:
                             break

                if i[1][0] == "b" and i[1][2:] in ["product", "area", "con", "pop"]:

                    inew.append(i[0])

                    for n1, i1 in enumerate(Preds[n + 1:]):
                        if i1[1][2:] == i[1][2:]:
                            inew.append(i1[0])

                        else:
                            break

                if inew:
                    if (inew, i[1]) not in Tokens and UntangleTokens(" ".join(inew), Tokens):
                        Tokens.append((inew, i[1]))
                    if i[1][2:].upper() not in Classes:
                        Classes.append(i[1][2:].upper())





            Tags = {

                "rd"      : "Outcomes / R&D",
                "product" : "Medicinal Product",
                "area"    : "Medical Condition",
                "str"     : "Strategy",
                "rem"     : "Regulatory Affairs",
                "con"     : "Connectors",
                "pop"     : "Patient Population",

                    }

            if [i for i in ["STR", "REM", "RD"] if i in Classes]:

                for Token in Tokens:

                    if Censor(Token[1], " ".join(Token[0])):

                        Text, Tag = st.beta_columns([5, 1.5])

                        with Text:

                            st.write(f"""<div style="font-size: 15px; font-family:Titillium Web; color:#125D98"><b>{" ".join(Token[0]).capitalize()}</b></div>""",
                                         unsafe_allow_html=True)
                        with Tag:

                            st.write(f"""<div style="font-size: 15px; font-family:Titillium Web; color:#F67280"><b>{Tags[Token[1][2:]]}</b></div>""",
                                         unsafe_allow_html=True)

            st.text("")
