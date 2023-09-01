import json

# Open the JSON file
with open("data.json", "r") as file:
    # Load the JSON data
    data = json.load(file)

print(data[1]["name"])
