from transformers import (AutoModelForTokenClassification,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          AutoTokenizer, 
                          pipeline,
                          )
from flask import Flask, request, jsonify, render_template
import os, openai
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings('ignore')

openai.api_key = os.getenv('OPENAI_KEY')

app = Flask(__name__)

print("loading classifier model")
model_dir = r".\distilbert-base-uncased-ade-cl.pt"
classify_tokenizer = DistilBertTokenizer.from_pretrained(model_dir,local_files_only=True)
classify_model = DistilBertForSequenceClassification.from_pretrained(model_dir,local_files_only=True)
classify_model = classify_model.to('cpu')
classifier = pipeline("text-classification", model=classify_model,tokenizer=classify_tokenizer)


print("loading ner model")
model_name = r".\ade_ner_model"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner = pipeline(task="ner", model=model, tokenizer=tokenizer)

prompt_template = '''Role: You are a medical assistant having knowledge of Medical Drugs and their adverse effects.
Job Description: Below statement and information related to that statement have been given to you which you will analyse as per your medical knowledge. If the information directs to an adverse drug effect with good confidence score then you will answer what is the adverse effect, which drug may be causing it, what is the recommended dosage of those drugs and if the patient should seek doctor consultation. You will also mention what kind of specialist doctor the patient should visit (if any). If the statement does not direct to any adverse drug effect or does so with a low Confidence score, then you will answer that this is not an adverse drug effect and explain why is it so.
Statement: {statement}
Drugs Mentioned: {drugs} 
Effects Mentioned: {effects}
Adverse Drug Effect Detected: {ade} with Confidence Score: {ade_score}

Answer:
'''

def find_drug_effect(json_list):
    drugs = []
    effects = []
    drug_temp = ''
    effect_temp = ''
    for i in range(len(json_list)):
        if json_list[i]['entity'] == 'B-DRUG':
            drug_temp = json_list[i]['word']
            for j in range(i+1, len(json_list)):
                if json_list[j]['entity'] == 'I-DRUG':
                    if json_list[j]['word'][0] == "#":
                      drug_temp += json_list[j]['word'][2:]
                    else:
                      drug_temp += " " + json_list[j]['word']
                else:
                    break
            drugs.append(drug_temp)
            drug_temp = ''
        elif json_list[i]['entity'] == 'B-EFFECT':
            effect_temp = json_list[i]['word']
            for j in range(i+1, len(json_list)):
                if json_list[j]['entity'] == 'I-EFFECT':
                    if json_list[j]['word'][0] == "#":
                      effect_temp += json_list[j]['word'][2:]
                    else:
                      effect_temp += " " + json_list[j]['word']
                else:
                    break
            effects.append(effect_temp)
            effect_temp = ''
    return drugs, effects

@app.route('/main')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_input():

    try:
        text = request.json['input_text']    

        print("getting classifier output")
        classifier_output= classifier(text)
        ade_output = classifier_output[0]['label']
        ade_confidence = classifier_output[0]['score']
        ade_output = 'Yes' if ade_output == 'Related' else 'No'

        print("getting ner output")
        ner_output = ner(text)
        drug_list, effect_list = find_drug_effect(ner_output)
        drugs_mentioned = ','.join(drug_list)
        effects_mentioned = ','.join(effect_list)

        filled_prompt = prompt_template.format(statement=text,
                                        drugs= drugs_mentioned if len(drugs_mentioned)>0 else 'Not Mentioned',
                                        effects= effects_mentioned if len(effects_mentioned)>0 else 'Not Mentioned',
                                        ade=ade_output,
                                        ade_score= ade_confidence)
        print("Input:\n",filled_prompt)

        response = openai.Completion.create(model="text-davinci-003",
                                        prompt=filled_prompt,
                                        temperature=0.3,
                                        max_tokens=2000,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0)

        output = response['choices'][0]['text']

        return jsonify({'status':200,
                        'drugs': drugs_mentioned if len(drugs_mentioned)>0 else 'Not Mentioned',
                        'effects': effects_mentioned if len(effects_mentioned)>0 else 'Not Mentioned',
                        'ade_output': ade_output,
                        'text_output': output})

    except Exception as e:
        return jsonify({'status': 403, 'message': f"Error: {str(e)}"})
    

if __name__ == '__main__':
    app.run(debug=True, host='localhost',port='5000')