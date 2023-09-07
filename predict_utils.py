from typing import Tuple, List


def symptoms_to_encoded_array(symptoms, symptom_index_dict):
    encoded_array = [0] * len(symptom_index_dict)
    for symptom in symptoms:
        if symptom in symptom_index_dict:
            index = symptom_index_dict[symptom]
            encoded_array[index] = 1
    return encoded_array


def predict_diseases(symptoms, model, symptom_index_dict, disease_labels) -> List[Tuple[str, float]]:
    encoded_symptoms = symptoms_to_encoded_array(symptoms, symptom_index_dict)
    probabilities = model.predict([encoded_symptoms])[0]
    return list(sorted(zip(disease_labels, probabilities), key=lambda x: x[1], reverse=True))[:5]
