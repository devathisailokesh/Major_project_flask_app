import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from azure.ai.textanalytics import TextAnalyticsClient , ExtractSummaryAction
from azure.core.credentials import AzureKeyCredential
from cleantext.sklearn import CleanTransformer


language_key = '5fc6758a908a4a9b9f3146b23afabb1b'
language_endpoint = 'https://angular.cognitiveservices.azure.com/'


def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential)
    return text_analytics_client


client = authenticate_client()
app = Flask(__name__)
CORS(app)


@app.route('/sem', methods=['POST'])
def index():
    documents=[request.data.decode('UTF-8')]
    
    result = client.analyze_sentiment(documents, show_opinion_mining=True)
    doc_result = [doc for doc in result if not doc.is_error]
    for document in doc_result:
        print("Document Sentiment: {}".format(document.sentiment))
        final = ("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
            document.confidence_scores.positive,
            document.confidence_scores.neutral,
            document.confidence_scores.negative,
        ))
    print(final)
    return jsonify({'msg': 'success', 'result': [final]})



@app.route('/sum', methods=['POST'])
def index1():
    temp = [request.data.decode('UTF-8')]
    cleaner = CleanTransformer(no_punct=False, lower=False)
    documents = cleaner.transform(temp)
    poller = client.begin_analyze_actions( documents, actions=[ExtractSummaryAction(),],)
    document_results = poller.result()
    for extract_summary_results in document_results:
        for result in extract_summary_results:
            final = ("Summary extracted: \n{}".format( " ".join([sentence.text for sentence in result.sentences])))
    print(final)
    return jsonify({'msg': 'success', 'result': [final]})
    



if __name__ == '__main__':
    app.run()
