""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

with open("../final_project/final_project_dataset.pkl", "rb") as file:
    enron_data = joblib.load(file)
    
    print(f"Number of entries: {len(enron_data.keys())}")
    
    num_features = 0
    for val in enron_data['METTS MARK'].values():
        num_features += 1
    print(f"Number of features: {num_features}")
    
    poi = 0
    for val in enron_data.values():
      if val.get('poi'):
          poi += 1
    
    print(f"Number of POI: {poi}")
    

def get_embedded_value(dictionary, key1, key2):
    return dictionary.get(key1).get(key2)

prentice_stock = get_embedded_value(dictionary=enron_data, key1='PRENTICE JAMES', key2='total_stock_value')

print(f"\nPrentice total stock: {prentice_stock}")

colwell_messages = get_embedded_value(enron_data, 'COLWELL WESLEY', 'from_this_person_to_poi')

print(f"Number of emails from Colwell to POI: {colwell_messages}")

skilling_exercised_options = get_embedded_value(enron_data, 'SKILLING JEFFREY K', 'exercised_stock_options')

print(f"Number of stcok options exercised by Skilling: {skilling_exercised_options}")
    
skilling_compensation = get_embedded_value(enron_data, 'SKILLING JEFFREY K', 'total_payments')

fastow_compensation = get_embedded_value(enron_data, 'FASTOW ANDREW S', 'total_payments')

lay_compensation = get_embedded_value(enron_data, 'LAY KENNETH L', 'total_payments')

print(f"Skilling Compensation: {skilling_compensation}")
print(f"Fastow Compensation: {fastow_compensation}")
print(f"Lay Compensation: {lay_compensation}")

print(f"Max Compensation: {max([lay_compensation, fastow_compensation, skilling_compensation])}")


salaries = sum(1 for val in enron_data.values() if val.get('salary') != 'NaN')
known_addresses = sum(1 for val in enron_data.values() if val.get('email_address') != 'NaN')

print(f"\nNumber of known Salaries: {salaries}")
print(f"Number of known emails : {known_addresses}")

nan_payments = 0
for val in enron_data.values():
    if isinstance(val.get('total_payments'), str):
        nan_payments += 1
        
print(f"NaN total payments: {nan_payments}")
print(f"Percentage NaN payments: {nan_payments/len(enron_data.keys()):.3f}")

nan_payments_alt = sum(1 for val in enron_data.values() if isinstance(val.get('total_payments'), str))


nan_payments_poi = 0

for val in enron_data.values():
    if val.get('poi') and isinstance(val.get('total_payments'), str):
        nan_payments_poi += 1

with open("../final_project/poi_names.txt", "r") as file:
    lines = file.readlines()[1:]
    
    print(f"\nHome POIs Exist? {len(lines)}")


    
