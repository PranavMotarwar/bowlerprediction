
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'playerid':2, 'wickettwo':1, 'wicketthreefour':0, 'wicketfive':0, 'economyrate':1, 'manofthematch':0, 'matchtype':3, 'inningstype':0})

print(r.json())