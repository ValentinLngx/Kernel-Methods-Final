import pandas as pd

Xte0 = pd.read_csv("data/Xte0.csv")
Xtr0 = pd.read_csv("data/Xtr0.csv")
Ytr0 = pd.read_csv("data/Ytr0.csv",dtype=float).drop(columns='Id').to_numpy()

Xte1 = pd.read_csv("data/Xte1.csv")
Xtr1 = pd.read_csv("data/Xtr1.csv")
Ytr1 = pd.read_csv("data/Ytr1.csv",dtype=float,delimiter=',').drop(columns='Id').to_numpy()

Xte2 = pd.read_csv("data/Xte2.csv")
Xtr2 = pd.read_csv("data/Xtr2.csv")
Ytr2 = pd.read_csv("data/Ytr2.csv",dtype=float).drop(columns='Id').to_numpy()

# data = np.loadtxt('file.csv', delimiter=',')

Xte0_mat100 = pd.read_csv("data/Xte0_mat100.csv", header=None, sep='\s+', engine='python',dtype=float).to_numpy()
Xtr0_mat100 = pd.read_csv("data/Xtr0_mat100.csv", header=None, sep='\s+', engine='python',dtype=float).to_numpy()
Xte1_mat100 = pd.read_csv("data/Xte1_mat100.csv", header=None, sep='\s+', engine='python',dtype=float).to_numpy()
Xtr1_mat100 = pd.read_csv("data/Xtr1_mat100.csv", header=None, sep='\s+', engine='python',dtype=float).to_numpy()
Xte2_mat100 = pd.read_csv("data/Xte2_mat100.csv", header=None, sep='\s+', engine='python',dtype=float).to_numpy()
Xtr2_mat100 = pd.read_csv("data/Xtr2_mat100.csv", header=None, sep='\s+', engine='python',dtype=float).to_numpy()
