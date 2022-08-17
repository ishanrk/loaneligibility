import numpy as np
import networkx as nx
import warnings 
import matplotlib.pyplot as plt
import matplotlib as mpl
from causalnex.structure import StructureModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import pandas as pd
from itertools import chain
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from sklearn.model_selection import train_test_split
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork

from causalnex.discretiser import Discretiser

warnings.filterwarnings("ignore")  # silence warnings

df = pd.read_csv('Loan_data.csv') # loading the dataset

df.drop(labels ='Loan_ID',axis = 1) # removing irrelevant features

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #creating a testset and trainset

#filtering data into continuous catgeorical or binary
cont_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
bin_cols = ['Gender','Married','Education','Self_Employed','Credit_History','Loan_Status']
cat_cols = ['Dependents','Loan_Amount_Term','Property_Area']

#encoders for transforming features

enc = LabelEncoder()  
minmax = MinMaxScaler()

for x in bin_cols:
    df[x] = enc.fit_transform(df[x])

for x in cat_cols:
    df[x] = enc.fit_transform(df[x])

for x in cont_cols:
    median = df[x].median() # option 3
    df[x].fillna(median, inplace=True)
   
df[cont_cols] = minmax.fit_transform(df[cont_cols])

#creating discretised copies

discretised_data = df.copy()
discretised_data.drop("Loan_ID",inplace=True, axis = 1)
print(discretised_data.head)
discretised_data["Loan_Amount_Term"] = Discretiser(method="fixed",numeric_split_points=[2,4,6]).transform(discretised_data["Loan_Amount_Term"].values)

discretised_data["CoapplicantIncome"] = Discretiser(method="fixed",numeric_split_points=[0.2,0.5,0.8]).transform(discretised_data["CoapplicantIncome"].values)
discretised_data["ApplicantIncome"] = Discretiser(method="fixed",numeric_split_points=[0.2,0.5,0.8]).transform(discretised_data["ApplicantIncome"].values)
discretised_data["LoanAmount"] = Discretiser(method="fixed",numeric_split_points=[0.2,0.5,0.8]).transform(discretised_data["LoanAmount"].values)

from causalnex.structure.pytorch import from_pandas

# converting the dataset intoa  structured model
sm = from_pandas(discretised_data, lasso_beta=1e-5, w_threshold=0, use_bias=True)
sm.threshold_till_dag()

Term_map = {0: "[0,36] months", 1: "(36,84] months", 2: "(84,180] months", 3:">180 months"}
AppIncome_map = {0: "Low-Income", 1: "Low-Average-Income", 2: "High-Average-Income", 3:"High-Income"}
CoappIncome_map = {0: "Low-Income", 1: "Low-Average-Income", 2: "High-Average-Income", 3:"High-Income"}
LoanAmount_map = {0: "Low-Amount", 1: "Low-Average-Amount", 2: "High-Average-Amount", 3:"High-Amount"}

discretised_data["Loan_Amount_Term"] = discretised_data["Loan_Amount_Term"].map(Term_map)
discretised_data["ApplicantIncome"] = discretised_data["ApplicantIncome"].map(AppIncome_map)
discretised_data["CoapplicantIncome"] = discretised_data["CoapplicantIncome"].map(CoappIncome_map)
discretised_data["LoanAmount"] = discretised_data["LoanAmount"].map(LoanAmount_map)

#plotting the graphy

fig = plt.figure(figsize=(15, 8))  # set figsize
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor("#001521")  # set backgrount

### run a layout algorithm to set the position of nodes
pos = nx.drawing.layout.circular_layout(sm) # various layouts available


# We can change the position of specific nodes


# add nodes to figure
nx.draw_networkx_nodes(
    sm,
    pos,
    node_shape="H",
    node_size=1000,
    linewidths=3,
    edgecolors="#4a90e2d9",
    node_color=["black" if "Cost" not in el else "#DF5F00" for el in sm.nodes],
)
# add labels
nx.draw_networkx_labels(
    sm,
    pos,
    font_color="#FFFFFFD9",
    font_weight="bold",
    font_family="Helvetica",
    font_size=10,
)
# add edges
nx.draw_networkx_edges(
    sm,
    pos,
    edge_color="white",
    node_shape="H",
    node_size=2000,
    
    width= 0.5
)

plt.show()

bn = BayesianNetwork(sm)    

from sklearn.model_selection import train_test_split

train, test = train_test_split(discretised_data, train_size=0.8, test_size=0.2, random_state=7)

bn = bn.fit_node_states(discretised_data)

bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

print(bn.cpds["Loan_Status"])
        
