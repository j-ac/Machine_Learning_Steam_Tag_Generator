import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

data = [
{'co-op': 1,    'FPS': 1,   'hordes': 1, 'warfare': 0,  'farm': 0,  'life': 0,  'adventure': 0}, # Deep Rock Galactic
{'co-op': 0,    'FPS': 1,   'hordes': 0, 'warfare': 1,  'farm': 0,  'life': 0,  'adventure': 0}, # Battlefield 2042
{'co-op': 1,    'FPS': 1,   'hordes': 0, 'warfare': 0,  'farm': 0,  'life': 0,  'adventure': 0}, # Battlefield 2042
{'co-op': 1,    'FPS': 0,   'hordes': 0, 'warfare': 0,  'farm': 1,  'life': 1,  'adventure': 0}, # Stardew Valley
{'co-op': 0,    'FPS': 0,   'hordes': 0, 'warfare': 0,  'farm': 1,  'life': 0,  'adventure': 0}, # Rune Factory 5
{'co-op': 0,    'FPS': 0,   'hordes': 0, 'warfare': 0,  'farm': 1,  'life': 1,  'adventure': 1}, # Kynseed
] 

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(data)
Y = np.array(['Shooter', 'Shooter', 'Shooter', 'Not Shooter', 'Not Shooter', 'Not Shooter'])

mnb = MultinomialNB()
mnb.fit(X, Y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

test_data = data = [
{'co-op': 0,    'FPS': 1,   'hordes': 0, 'warfare': 0,  'farm': 0,  'life': 0,  'adventure': 0},  
{'co-op': 1,    'FPS': 0,   'hordes': 0, 'warfare': 0,  'farm': 1,  'life': 1,  'adventure': 0}, 
{'co-op': 0,    'FPS': 1,   'hordes': 0, 'warfare': 1,  'farm': 0,  'life': 0,  'adventure': 0}, 
{'co-op': 1,    'FPS': 1,   'hordes': 0, 'warfare': 0,  'farm': 0,  'life': 0,  'adventure': 0}, 
{'co-op': 0,    'FPS': 1,   'hordes': 0, 'warfare': 0,  'farm': 0,  'life': 0,  'adventure': 0}, 
{'co-op': 1,    'FPS': 0,   'hordes': 1, 'warfare': 0,  'farm': 0,  'life': 0,  'adventure': 0}, 
{'co-op': 0,    'FPS': 0,   'hordes': 0, 'warfare': 1,  'farm': 0,  'life': 1,  'adventure': 1}, 
{'co-op': 0,    'FPS': 1,   'hordes': 0, 'warfare': 0,  'farm': 1,  'life': 0,  'adventure': 0}, 
]

print(mnb.predict(dv.fit_transform(test_data)))