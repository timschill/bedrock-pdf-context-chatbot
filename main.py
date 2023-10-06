import io
import re
import os
import boto3
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def convert_pdf_to_txt(path):
    resource_manager = PDFResourceManager()
    device = None
    try:
        with io.StringIO() as string_writer, open(path, "rb") as pdf_file:
            device = TextConverter(
                resource_manager, string_writer, laparams=LAParams(line_margin=0.1)
            )
            interpreter = PDFPageInterpreter(resource_manager, device)

            for page in PDFPage.get_pages(pdf_file, maxpages=0):
                interpreter.process_page(page)

            pdf_text = string_writer.getvalue()
    finally:
        if device:
            device.close()
    return pdf_text


def bedrock_chain(pdf_text):
    profile = os.environ["AWS_PROFILE"]

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    model_kwargs = {
        "maxTokenCount": 300,
        "temperature": 0,
    }

    chat = Bedrock(
        credentials_profile_name=profile,
        model_id="amazon.titan-text-express-v1",
        model_kwargs=model_kwargs,
        client=bedrock_runtime,
    )

    template = """The following is a friendly conversation between a knowledgeable helpful AI and a human.
The AI is talkative and provides lots of specific details from it's context included between the <document> and </document> tags.
If the AI does not know the answer to a question, it truthfully says it does not know.

<document>
{context}
</document>

Conversation:
    """

    smpt = SystemMessagePromptTemplate.from_template(template)
    smpt = smpt.format(context=pdf_text)
    messages = [
        smpt,
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(
        human_prefix="Human", ai_prefix="AI", return_messages=True
    )
    conversation = ConversationChain(
        prompt=qa_prompt, llm=chat, verbose=True, memory=memory
    )

    return conversation


def run_chain(chain, prompt):
    return chain({"input": prompt})


def clear_memory(chain):
    return chain.memory.clear()


if __name__ == "__main__":
    formatted_text = re.sub(r"[^a-zA-Z0-9 \n\.]", "", convert_pdf_to_txt("test.pdf"))
    removed_excessive_newlines = re.sub(r"\n\s*\n", "\n\n", formatted_text)

    chain = bedrock_chain(removed_excessive_newlines)
    resp = chain.predict(input="Hello there!")
    print(resp)
