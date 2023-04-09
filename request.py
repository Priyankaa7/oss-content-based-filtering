import requests

url = 'http://localhost:5000/oss_api'
r = requests.post(url,json={'name':100})

print(r.json())