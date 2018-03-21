# ------------- Author -------------#
# Samreen A. Khan

# ------------- Description ------------#
# Developing a recommendation system using movies data
# Twon types of system
# Content based 
# Collaborative


import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetching the data and format it - fetching only that data that has the rating >= 4
data = fetch_movielens(min_rating = 4.0)

# Print the training and testing dataset by creating dictionary
print(repr(data['train']))
print(repr(data['test']))

# Creating model
# warp = Weighted Approximate-Rank Pairwise
# WARP uses Gradient Descent
model = LightFM(loss='warp') # Loss means the difference between Model prediction and desire output

# Training the model
model.fit(data['train'], epochs = 30, num_threads = 2)

def sample_recommendation(model, data, user_ids):
    
    # setting the numbe rof users & movies in training data
    no_user, np_item = data['train'].shape
    
    # Genertaing recommendatopns for each user we input
    for user_id in user_ids:
        
        # Movies that user already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        # Movies that this model will predict
        scores = model.predict(user_id, np.arange(n_items))
        
        # Rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        
        # Results
        print("User %s" % user_id)
        print("     Known positives: ")
        
        for x in known_positives[:3]:
            print("     %s" % x)
            
        print("    Recommended:")
        
        for x in top_items[:3]:
            print("     %s" % x)
            
sample_recommendation(model, data, [3, 25, 458])
        
        
        
        
        
        
        
        
        
        
        