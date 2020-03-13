import json

name = 'kushman'
prefix = 'data/microsoft/'
data_path = prefix + name + '.json'

with open(data_path, 'r') as file:
    data = json.load(file)

print(len(data))
train_proportion = 0.8
cutoff = int(len(data) * train_proportion)
train_data = data[:cutoff]
val_data = data[cutoff:]

with open(prefix + name + '_train.json', 'w') as file:
    json.dump(train_data, file)

with open(prefix + name + '_val.json', 'w') as file:
    json.dump(train_data, file)
