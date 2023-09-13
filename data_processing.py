import base64
import pandas as pd
from prefixspan import PrefixSpan
from mlxtend.frequent_patterns import apriori, association_rules
from gsppy.gsp import GSP
from mlxtend.frequent_patterns import fpgrowth


def tranform_set(x):
    return ', '.join(list(x))

def run_apriori(medical_condition_df, user_min_sup, user_min_confidence, user_min_pattern_length, user_excluded_features):
    # Convert user_min_confidence from percentage to float between 0 and 1
    user_min_sup = user_min_sup / 100.00 
    user_min_confidence = user_min_confidence / 100.00
    
    #Apply Apriori
    frequent_itemsets = apriori(medical_condition_df, min_support=user_min_sup, use_colnames=True)

    #Generate association rules 
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=user_min_confidence)

    # Filter rules based on pattern length and excluded features
    patterns_df = rules[
        ~rules.apply(lambda row: len(row['antecedents']) + len(row['consequents']) < user_min_pattern_length or any(feature in row['antecedents'] for feature in user_excluded_features), axis=1)
    ]

    #Rename columns for clarity
    patterns_df = patterns_df.rename(columns={'antecedents': 'Antecedents', 'consequents': 'Consequents', 'support': 'Support', 'confidence': 'Confidence'})

    # Convert 'Support' and 'Confidence' to percentages
    patterns_df['Support'] = (patterns_df['Support'] * 100).round(2).astype(str) + '%'
    patterns_df['Confidence'] = (patterns_df['Confidence'] * 100).round(2).astype(str) + '%'

    # Create the 'Pattern' column by merging 'Antecedents' and 'Consequents'
    patterns_df['Pattern'] = patterns_df.apply(lambda row: f"{', '.join(row['Antecedents'])} -> {', '.join(row['Consequents'])}", axis=1)

    # Select and reorder the desired columns
    patterns_df = patterns_df.loc[:, ['Pattern', 'Support', 'Confidence']]

    return patterns_df

def run_fpgrowth(medical_condition_df, user_min_sup, user_min_confidence, user_min_pattern_length, user_excluded_features):
    # Convert user_min_confidence from percentage to float between 0 and 1
    user_min_sup = user_min_sup / 100.00 
    user_min_confidence = user_min_confidence / 100.00
    
    #Apply Apriori
    frequent_itemsets = fpgrowth(medical_condition_df, min_support=user_min_sup, use_colnames=True)

    #Generate association rules 
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=user_min_confidence)

    # Filter rules based on pattern length and excluded features
    patterns_df = rules[
        ~rules.apply(lambda row: len(row['antecedents']) + len(row['consequents']) < user_min_pattern_length or any(feature in row['antecedents'] for feature in user_excluded_features), axis=1)
    ]

    #Rename columns for clarity
    patterns_df = patterns_df.rename(columns={'antecedents': 'Antecedents', 'consequents': 'Consequents', 'support': 'Support', 'confidence': 'Confidence'})

    # Convert 'Support' and 'Confidence' to percentages
    patterns_df['Support'] = (patterns_df['Support'] * 100).round(2).astype(str) + '%'
    patterns_df['Confidence'] = (patterns_df['Confidence'] * 100).round(2).astype(str) + '%'

    # Create the 'Pattern' column by merging 'Antecedents' and 'Consequents'
    patterns_df['Pattern'] = patterns_df.apply(lambda row: f"{', '.join(row['Antecedents'])} -> {', '.join(row['Consequents'])}", axis=1)

    # Select and reorder the desired columns
    patterns_df = patterns_df.loc[:, ['Pattern', 'Support', 'Confidence']]

    return patterns_df

def run_gsp(patterns_gsp, user_min_sup, user_min_confidence, user_min_pattern_length, user_excluded_features):
    # Convert user_min_confidence from percentage to float between 0 and 1
    user_min_sup = user_min_sup / 100.00 
    user_min_confidence = user_min_sup / 100.00 
    
    # Apply GSP:
    ps = GSP(patterns_gsp)
    results = ps.search(user_min_sup)
    total_transactions = len(patterns_gsp)

    filtered_results = []
    #Loop through dictionaries of which index >= min_pattern_length-1
    for dictionary in results[user_min_pattern_length-1::]:
        for key, occurences in dictionary.items():
            # Exclude features:
            if all(elem not in user_excluded_features for elem in key):
                pattern = ', '.join(key)
            # Calculate support 
                support = (occurences / total_transactions) 
            # Calculate Confidence:
                antecedent = tuple(key[:-1])
                if antecedent in results[user_min_pattern_length - 2]:
                    antecedent_support = results[user_min_pattern_length - 2][antecedent]
                    confidence = (occurences / antecedent_support)
                    if confidence >= user_min_confidence:
                        filtered_results.append((pattern, support, confidence))

    # Sort the results according to pattern occurences:    
    sorted_filtered_results = sorted(filtered_results, key=lambda x: x[0], reverse=True)
    
# Convert 'Support' and 'Confidence' to percentages
    patterns_df = pd.DataFrame(sorted_filtered_results, columns=['Pattern','Support','Confidence'])
    patterns_df['Support'] = (patterns_df['Support'] * 100).round(2).astype(str) + '%'   
    patterns_df['Confidence'] = (patterns_df['Confidence'] * 100).round(2).astype(str) + '%'   

    return patterns_df
    
def map_indexes_to_action_names(pattern, action_names):
    action_names_str = [action_names[i] for i in pattern]
    return ", ".join(action_names_str)

def run_prefixspan(patterns, user_min_sup, user_min_confidence, user_min_pattern_length, user_excluded_features, name_list):
    #Apply Prefix Span
    total_patterns = len(patterns)
    user_min_sup = (user_min_sup / 100.00) * total_patterns
    user_min_confidence = user_min_confidence / 100.00

    ps = PrefixSpan(patterns)
    results = ps.frequent(int(user_min_sup))
    
    #Get index of excluded features:
    indexes_excluded_features = [name_list.index(name) for name in user_excluded_features]

   # Calculate Confidence for each pattern and filter by user_min_confidence
    filtered_results = []

    for pattern in results:
        support = pattern[0]
        pattern = pattern[1]
        
        if len(pattern) >= user_min_pattern_length:
            antecedent = tuple(pattern[:-1])
            antecedent_support = 0
            
            for p in results:
                if tuple(p[1]) == antecedent:
                    antecedent_support = p[0]
                    break
            
            if antecedent_support > 0:
                confidence = float((support / antecedent_support))
                if confidence >= user_min_confidence:
                  # Map indexes to action names
                  action_names = map_indexes_to_action_names(pattern, name_list)
                  filtered_results.append((action_names, support, confidence))

    # Sort the results based on support
    sorted_filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

    # Create a DataFrame with the results and convert support to percentage
    patterns_df = pd.DataFrame(sorted_filtered_results, columns=['Pattern', 'Support', 'Confidence'])
    patterns_df['Support'] = ((patterns_df['Support'] / total_patterns) * 100.00).round(2).astype(str) + '%'
    patterns_df['Confidence'] = (patterns_df['Confidence'] * 100).round(2).astype(str) + '%'   

    return patterns_df

