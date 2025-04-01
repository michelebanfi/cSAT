from mixed import generate_mixed_causal_dataframe

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from causallearn.utils.GraphUtils import GraphUtils

df = generate_mixed_causal_dataframe()

cg = pc(df.to_numpy(), indep_test=kci, kernelZ="Polynomial", show_progress=True)

names = list(df.columns)
# for i, node in enumerate(cg.G.nodes):
#     node.name = names[i]

pyd = GraphUtils.to_pydot(cg.G, labels=names)
pyd.write_png('miscellanous/KCI.png')

print(df)