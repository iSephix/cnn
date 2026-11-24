import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from card import compile_tree,predict

for seed in range(12):
    rng=np.random.default_rng(seed)
    x=rng.integers(-1,13,(150,10))
    y=np.where(x[:,3]<4,10,np.where(x[:,0]>6,7,3)) if seed else np.full(150,10)
    m=DecisionTreeClassifier(random_state=0).fit(x,y)
    card=json.loads(json.dumps(compile_tree(m)))
    fresh=rng.integers(-3,16,(500,10))
    assert np.array_equal(predict(card,fresh),m.predict(fresh))
print('6,000 serialized-card predictions matched the reference tree.')
