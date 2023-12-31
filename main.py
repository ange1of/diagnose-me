import json

from fastapi import FastAPI
from Levenshtein import setratio
from starlette.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

from models import (
    SymptomList,
    SearchSymptomQuery,
)
from predict_utils import predict_diseases

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('symptoms_dictionary.json', 'r', encoding='utf-8') as symptoms_file:
    SYMPTOMS = json.load(symptoms_file)

FLAT_SYMPTOMS = {}
for category, category_data in SYMPTOMS.items():
    for symptom_ru, symptom_en in category_data.items():
        FLAT_SYMPTOMS[symptom_ru] = symptom_en

DIAGNOSE_MODEL = load_model('hahatonV2.h5')

with open('disease_labels.json', 'r') as json_file:
    DISEASE_LABELS = json.load(json_file)

DISEASE_TRANSLATIONS = {}
with open('diseases_translations.json', 'r', encoding='utf-8') as json_file:
    for disease_ru, disease_en in json.load(json_file).items():
        DISEASE_TRANSLATIONS[disease_en] = disease_ru

with open('symptom_index_dict.json', 'r') as json_file:
    SYMPTOM_INDEX_DICT = json.load(json_file)


@app.post('/predict')
def predict(symptom_list: SymptomList):
    if not symptom_list.symptoms:
        raise ValueError('No symptoms provided')

    symptoms = [s for s in symptom_list.symptoms if s in FLAT_SYMPTOMS.values()]

    result = predict_diseases(symptoms, DIAGNOSE_MODEL, SYMPTOM_INDEX_DICT, DISEASE_LABELS)

    return [
        {'disease': DISEASE_TRANSLATIONS.get(disease), 'probability': round(probability * 100, 1)}
        for disease, probability in result
    ]


@app.post('/search-symptom')
def search_symptom(search_query: SearchSymptomQuery):
    if not search_query.query:
        return ''

    query_list = search_query.query.lower().split()

    return list(
        sorted(
            FLAT_SYMPTOMS.items(),
            key=lambda item: setratio(item[0].lower().split(), query_list),
            reverse=True,
        )
    )[:5]
