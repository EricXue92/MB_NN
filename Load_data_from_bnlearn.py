# https://stackoverflow.com/questions/59474686/dataset-for-bayesian-network-structure-learning
import bnlearn as bn

# bif_file= 'sprinkler'
# bif_file= 'alarm'
# bif_file= 'andes'
# bif_file= 'asia'
# bif_file= 'pathfinder'
# bif_file= 'sachs'
# bif_file= 'miserables'


bif_file= 'Raw_DATASET/insurance.bif'

# Loading DAG with model parameters from bif file.
model = bn.import_DAG(bif_file)

print(type(model))

df = bn.sampling(model, n=2000, methodtype='bayes')
print(df.head())