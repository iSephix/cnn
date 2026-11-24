"""Compile an sklearn decision tree into a JSON-serializable threshold network."""
import numpy as np

def compile_tree(model):
    tree=model.tree_
    predicates=[]
    leaves=[]
    def visit(node,path):
        if tree.children_left[node]==tree.children_right[node]:
            color=int(model.classes_[tree.value[node].argmax()])
            leaves.append([path,color])
            return
        idx=len(predicates)+1
        predicates.append([int(tree.feature[node]),float(tree.threshold[node])])
        visit(tree.children_left[node],path+[-idx])
        visit(tree.children_right[node],path+[idx])
    visit(0,[])
    return {'format':'arc-cell-threshold-v1','predicates':predicates,'leaves':leaves}

def predict(card,features):
    # Gates have the same float32 inputs and strict-right convention as training.
    gates=[features[:,j].astype(np.float32)>t for j,t in card['predicates']]
    out=np.full(len(features),-1)
    for path,color in card['leaves']:
        hits=np.zeros(len(features),dtype=int)
        for literal in path:
            gate=gates[abs(literal)-1]
            hits+=gate if literal>0 else ~gate
        out[hits==len(path)]=color
    if (out<0).any():raise ValueError('Card failed to cover its inputs')
    return out
