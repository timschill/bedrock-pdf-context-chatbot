## bedrock pdf context chatbot
This simple application will extract the text from a PDF file and insert it into a prompt template.  
You can then query the Amazon Bedrock Fondation model (FM) with questions in regards to the content inside the PDF.

### How to get started
Simply export the AWS profile you want to work with from the terminal,
```
export AWS_PROFILE="profile_name"
```
You can change the pdf file you want to work with here
```
formatted_text = re.sub(r"[^a-zA-Z0-9 \n\.]", "", convert_pdf_to_txt("test.pdf"))
```
Install the dependencies, it is recommended to do this in a virtual environment.
```
pip install -r requirements.txt
```