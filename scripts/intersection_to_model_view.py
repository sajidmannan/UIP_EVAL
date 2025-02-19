import json

# Load the original JSON data
with open('./mappings/intersection_mapping.json', 'r') as file:
    data = json.load(file)

# Initialize an empty dictionary to hold the new structure
nested_dict = {}

# Process the data to create the nested dictionary
for material, indices in data.items():
    for model, index in indices.items():
        if model not in nested_dict:
            nested_dict[model] = []
        nested_dict[model].append(index)

# Check if all models have lists of the same length
list_lengths = [len(indices) for indices in nested_dict.values()]
if len(set(list_lengths)) != 1:
    raise ValueError("Not all models have lists of the same length")

# Save the new nested dictionary to a new JSON file
with open('./mappings/intersection_mapping_by_model.json', 'w') as file:
    json.dump(nested_dict, file, indent=4)

print("Nested dictionary has been saved to intersection_mapping_by_model.json")