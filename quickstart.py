import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("""dataset_path""")
df.columns = ['user', 'traj', 'point', 'cell']
df['traj'] = pd.to_numeric(df['traj'], downcast='integer')
df['point'] = pd.to_numeric(df['point'], downcast='integer')
df['cell'] = pd.to_numeric(df['cell'], downcast='integer')

locations = pd.read_csv("""dataset_path""", header=None)
locations.columns = ['id', 'long', 'lat']

trajs = treemob.generate_trajs(df_user)
trajs = treemob.prepocess_trajs(trajs, locations.id.values, locations.long.values, locations.lat.values, 0.001)

tree = treemob.PrefixTree(trajs['trajs'], trajs['root'], nodes_weights=trajs['n_weights'], 
                  edges_weights=trajs['e_weights'], 
                  user=user, city='grosseto')
treemob.save_tree(tree, str(user))

