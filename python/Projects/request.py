import requests

res = requests.get("https://ec3.dev/", stream = True)
# print(res)

# print(res.text)

print(res.raw)

print(res.raw.read(1))