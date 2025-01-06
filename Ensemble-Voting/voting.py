
m = {'accuracy_most_likely': 0.9958991494532199,
 'precision_differential': 0.9725353996770867,
 'recall_differential': 0.9751223975634911,
 'f1_differential': 0.9702945402271386,
 'rouge_differential': 0.9448142731543822}


p ={'accuracy_most_likely': 0.9769,
'precision_differential': 0.9176261585108249, 
 'recall_differential': 0.9267435226579821, 
 'f1_differential': 0.9121278857091725, 
 'rouge_differential': 0.8773032723351539}

l = {'accuracy_most_likely': 0.82,
'precision_differential': 0.8, 
 'recall_differential': 0.7, 
 'f1_differential': 0.7, 
 'rouge_differential': 0.7}


# Relative Improvement= 1+ (Value−Min Value) / (1−Min Value)

models = {'mistral': m, 'phi': p, 'llama': l}

# Step 1: Find the minimum value for each metric (used as the baseline)
metrics = m.keys()
min_values = {metric: min(model[metric] for model in models.values()) for metric in metrics}

# Step 2: Calculate relative improvement for each model's metrics
relative_improvements = {}
for model_name, metrics_values in models.items():
    relative_improvements[model_name] = {
        metric: 1 + ((metrics_values[metric] - min_values[metric]) / (1 - min_values[metric]))
        for metric in metrics_values
    }

# Step 3: Compute total relative improvement score for each model
total_scores = {model_name: sum(metrics.values()) for model_name, metrics in relative_improvements.items()}
# Step 4: Normalize the total scores to get weights
total_sum = sum(total_scores.values())
weights = {model_name: total_score / total_sum for model_name, total_score in total_scores.items()}

print(weights)
outputs ={
        'mistral' :{
            'most_likely': 'AIDS',
            'differential': ['HIV (initial infection)', 'Anemia', 'Dengue' ]
        },
        'llama': {
            'most_likely': 'XYZ',
            'differential': ['HIV (initial infection)', 'Colitis', 'jaundice']
        },
        'phi': {
            'most_likely': 'Anemia',
            'differential': ['HIV (initial infection)', 'Anemia', 'Pancreatic neoplasm', 'Chostochondritis']
        },
}

most_likely = {}

for model, output in outputs.items():
    if output['most_likely'] in most_likely:
        most_likely[output['most_likely']] += 1
    else:
        most_likely[output['most_likely']] = 1

differential  = {} 
for model, output in outputs.items():
    for diagnosis in output['differential']:
        if diagnosis in differential:
            differential[diagnosis] += weights[model]
        else:
            differential[diagnosis] = weights[model]
    
differential_items = sorted(differential.items(), key=lambda x: x[1], reverse=True)
print(differential_items)

diagnosis =None
differential_diagnosis = []
for key, value in most_likely.items():
    if value>=2:
        diagnosis = key
        break
if not diagnosis:
    diagnosis = differential_items[0][0]

for idx, (key, value) in enumerate(differential_items):
    differential_diagnosis.append(key)
    if idx>=4:
        break

print(f'Differential diagnosis is: {differential_diagnosis} and the most likely diagnosis is {diagnosis}')
    
