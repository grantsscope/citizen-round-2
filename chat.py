import os
import streamlit as st
import langchain
#from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from trubrics.integrations.streamlit import FeedbackCollector
import random

langchain.verbose = True

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None

collector = FeedbackCollector(
    project="CR2",
    email=st.secrets["TRUBRICS_EMAIL"],
    password=st.secrets["TRUBRICS_PWD"],
)

# Initialize Streamlit app configuration
st.set_page_config(page_title="GrantsScope - Gitcoin Citizens Round II")
st.header('GrantsScope')
st.subheader ('Gitcoin Citizens Round #2')
st.markdown('[The Gitcoin Citizens Round](https://gov.gitcoin.co/t/rewarding-the-community-gitcoin-citizens-round-2/16506) supports individuals running new or ongoing projects that matter enormously to the Gitcoin community. Donations open until October 24th 23:59 UTC. GrantsScope is also a grantee in this round seeking retroactive funding.')

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.link_button("Explore Projects", "https://explorer.gitcoin.co/#/round/424/0x7492a8c4ed29b1f1559888bd832fae5d33e10370",type="primary")
with col2:
    st.link_button("Donate to GrantsScope", "https://explorer.gitcoin.co/#/round/424/0x7492a8c4ed29b1f1559888bd832fae5d33e10370/0x7492a8c4ed29b1f1559888bd832fae5d33e10370-47",type="secondary")

index = './storage/faiss_index'
embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vectors = FAISS.load_local(index, embeddings)

prompt_template = """We have provided context information below. 
---------------------
{context}
---------------------
Do not respond to questions that ask to sort or rank grantees. Do not respond to questions that ask to compare grantees. Similarly, do not respond to questions asking for advice on which grantee to donate contributions. Few examples of such questions are (a) Which grantee had the most impact on Gitcoin? (b) Who should I donate to? (c) Rank the grantees by impact (d) Compare work of one grantee versus another? For such questions, do not share any grantee information and just say: "Dear human, I am told not to influence you with my biases for such queries. The burden of choosing the public greats and saving the future of your kind lies on you. Choose well!"
If the answer is unavailable in the context information above, respond: Sorry! I don't have an answer for this.
Given this information, please answer the following question. Include Explorer Links when sharing grantee information. Respond in table format when there are more than 2 grantees in the response.
Question: {question}"""

prompt_type = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": prompt_type}


chain = ConversationalRetrievalChain.from_llm(
	llm = ChatOpenAI(
		temperature=0.0,
		model_name='gpt-3.5-turbo-16k'
		),
	retriever=vectors.as_retriever(),
	memory=memory,
	combine_docs_chain_kwargs=chain_type_kwargs,
	#max_tokens_limit=3000
	)

def conversational_chat(query):
    
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    
    return result["answer"]

# Trivia

trivia = [
    "Did you know that many prominent web3 organizations were once grantees of Gitcoin, and a large number have even returned as funders, like Push Protocol, Mask Network, 1inch.",
    "Have you checked out the GG19 Outline and Strategy yet? Join the discussion [here](https://gov.gitcoin.co/t/gg19-outline-and-strategy/16682)",
    "Did you know Grants Stack supports Direct Grants?! [Click](https://x.com/grantsstack/status/1701967342559694938?s=20) to hear more from Meg Lister, Product Lead.",
    "You can now move your Passport Onchain to access high quality, trustworthy onchain opportunities. Read more about it [here](https://www.gitcoin.co/blog/gitcoin-passport-onchain-stamps).",
    "What has the Gitcoin team got up to in September and what's ahead? Click [here](https://x.com/gitcoin/status/1707761192725336170?s=20) to find out.",
    "CharmVerse, an operations platform for web3 communities, was looking for authentic and value-creating ways to bring in new users. CharmVerse uses Allo to enable any group that registers to automatically get a claimable workspace for their project."
]


# Initialize chat history

st.session_state['history'] = []

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg="""Hi there 👋!""" 
    #st.chat_message("assistant").markdown(welcome_msg)
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("Ask me about the grantees in this round."):
	#History

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("While I look that up for you, here's some Gitcoin trivia: " + "\n\n" + random.choice(trivia))        
        response = conversational_chat(prompt)
        message_placeholder.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.logged_prompt = collector.log_prompt(
        config_model={"model": "gpt-3.5-turbo-16k"},
        prompt=prompt,
        generation=response,
    )


if st.session_state.logged_prompt:
    user_feedback = collector.st_feedback(
        component="CR2_FB",
        feedback_type="thumbs",
        open_feedback_label="[Optional] Provide additional feedback",
        model="gpt-3.5-turbo-16k",
        prompt_id=st.session_state.logged_prompt.id,
    )

    #if user_feedback:
    #     trubrics_successful_feedback(user_feedback)
