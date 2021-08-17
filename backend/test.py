import requests

BASE = "http://172.17.8.59:5000/"

text1 = "Geoffrey Everest Hinton (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks."
response = requests.put(BASE + "spacy", {'model': 'en_core_web_sm',
                                       'text': [text1],
                                       'mode': 'token'})
res = response.json()
print(res)

response = requests.put(BASE + "spacy", {'model': 'en_core_web_sm',
                                      'text': [text1],
                                      'mode': 'char'})
res = response.json()

print(res)
