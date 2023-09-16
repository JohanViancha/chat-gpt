import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from colorama import Back,Style,Fore, init
init()


lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())  # Cargamos un archivo JSON que contiene los patrones de preguntas y respuestas del chatbot.
words = pickle.load(open('words.pkl', 'rb'))  # Cargamos un archivo pickle que contiene una lista de palabras relevantes.
classes = pickle.load(open('classes.pkl', 'rb'))  # Cargamos un archivo pickle que contiene las clases o categorías de respuestas del chatbot.
model = load_model('chatbot_model.h5')  # Cargamos el modelo de red neuronal previamente entrenado para clasificar las preguntas.

# Pasamos las palabras de una oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenizamos la oración en palabras individuales.
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Aplicamos lematización a las palabras.
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Lematizamos y tokenizamos la oración.
    bag = [0] * len(words)  # Inicializamos una lista de ceros del mismo tamaño que la lista de palabras relevantes.
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Marcamos con 1 las palabras presentes en la oración.
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Convertimos la oración en una representación de bolsa de palabras.
    res = model.predict(np.array([bow]))[0]  # Usamos el modelo para predecir la categoría.
    max_index = np.where(res == np.max(res))[0][0]  # Encontramos la categoría con la probabilidad más alta.
    category = classes[max_index]  # Obtenemos la categoría correspondiente.

    return category

# Obtenemos una respuesta
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']  # Obtenemos la lista de patrones de preguntas y respuestas.
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])  # Seleccionamos una respuesta de las respuestas disponibles.
            break
    return result

# Ejecutamos el chat en bucle
while True:
    print( Fore.BLUE + 'Hazme una pregunta:')  
    print();  
    message = input("")  # Esperamos una entrada del usuario.
    print(Fore.BLUE + '=================================')
    ints = predict_class(message)  # Predecimos la categoría de la pregunta del usuario.
    res = get_response(ints, intents)  # Obtenemos una respuesta apropiada para la categoría.
    print(Style.BRIGHT + 'Respuesta del chatbot:')
    
    print(Fore.GREEN + res)  # Mostramos la respuesta al usuario.
    print()
