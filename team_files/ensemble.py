def ensemble_responses(outputs):
    """
    Ensembles responses by the 3 medical models
    """

    weights = {'mistral': 0.41213317355843804, 'phi': 0.3703593124949766, 'llama': 0.21750751394658532}
    
    # Find the most likely diagnosis with equal weightage for all models
    most_likely = {}
    for model, output in outputs.items():
        if output['most_likely'] in most_likely:
            most_likely[output['most_likely']] += 1
        else:
            most_likely[output['most_likely']] = 1
    
    # Find the differential diagnosis with weights
    differential  = {} 
    for model, output in outputs.items():
        for diagnosis in output['differential']:
            if diagnosis in differential:
                differential[diagnosis] += weights[model]
            else:
                differential[diagnosis] = weights[model]
    differential_items = sorted(differential.items(), key=lambda x: x[1], reverse=True)

    diagnosis =None
    differential_diagnosis = []
    for key, value in most_likely.items():
        # If a diagnosis has been predicted by atleast 2 models, then that is the most likely diagnosis
        if value>=2:
            diagnosis = key
            break
    if not diagnosis:
        # Else the most likely diagnosis is the one with the highest weightage in the differential diagnosis
        diagnosis = differential_items[0][0]

    # Get the top 5 differential diagnosis
    for idx, (key, value) in enumerate(differential_items):
        differential_diagnosis.append(key)
        if idx>=4:
            break
    differential_diagnosis = ', '.join(differential_diagnosis)
    final_response = f'Differential diagnosis is: {differential_diagnosis} and the most likely diagnosis is {diagnosis}'
    return final_response