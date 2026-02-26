# uri_agi_final_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="URI-AGI Mini Application", layout="wide")
st.title("Unified Relational Intelligence (URI-AGI) Mini App")

# -------------------------
# URI-AGI Core Classes
# -------------------------
class URIAgent:
    """Causal Reasoning Node using 24 SU(5) Generators"""
    def __init__(self):
        self.generators = self._init_generators()
        self.weights = np.random.dirichlet(np.ones(24), size=1)[0]

    def _init_generators(self):
        matrices = []
        # Diagonal Cartan (4)
        matrices.append(np.diag([1,-1,0,0,0]))
        matrices.append(np.diag([1,1,-2,0,0])/np.sqrt(3))
        matrices.append(np.diag([1,1,1,-3,0])/np.sqrt(6))
        matrices.append(np.diag([1,1,1,1,-4])/np.sqrt(10))
        # Step Operators + Interactions (20)
        for _ in range(20):
            matrices.append(np.random.randn(5,5))  # Placeholder
        return matrices

    def compute_causality(self, input_vector):
        causal_matrix = sum(w*G for w,G in zip(self.weights, self.generators))
        return np.dot(causal_matrix, input_vector)

class URICounselor(URIAgent):
    """Simplified Causal Counseling"""
    def counsel(self, user_input):
        diagnosis = "Upanissaya (Past Habitual Momentum) is high."
        remedy = "Apply Asevana (New Habitual Momentum) through Nissaya (Support)."
        return f"Causal Analysis: {diagnosis}\nSuggested Path: {remedy}"

# -------------------------
# Sidebar: Configuration
# -------------------------
st.sidebar.header("Simulation Settings")
num_nodes = st.sidebar.slider("Number of Agents", 1, 10, 3)
vector_dim = st.sidebar.slider("Input Vector Dimension", 5, 10, 5)
custom_input = st.sidebar.text_input("Optional Input Vector (comma-separated)", "")

# Generate input vector
if custom_input:
    try:
        input_vec = np.array([float(x) for x in custom_input.split(",")])
        if len(input_vec) != vector_dim:
            st.warning(f"Input length mismatch. Using random vector of dimension {vector_dim}.")
            input_vec = np.random.randn(vector_dim)
    except:
        st.warning("Invalid input. Using random vector.")
        input_vec = np.random.randn(vector_dim)
else:
    input_vec = np.random.randn(vector_dim)

st.subheader("Input Vector")
st.write(input_vec)

# -------------------------
# Initialize Agents
# -------------------------
agents = [URICounselor() for _ in range(num_nodes)]
outputs = np.array([agent.compute_causality(input_vec) for agent in agents])

# -------------------------
# Display Outputs
# -------------------------
st.subheader("Agent Outputs")
for i, out in enumerate(outputs):
    st.write(f"Agent {i+1} output:", out)

# -------------------------
# Visualization
# -------------------------
st.subheader("2D Projection of Agent Outputs")
pca = PCA(n_components=2)
proj = pca.fit_transform(outputs)
plt.figure(figsize=(6,4))
plt.scatter(proj[:,0], proj[:,1], c='blue', s=100)
for i, txt in enumerate(range(len(proj))):
    plt.annotate(f"Agent {i+1}", (proj[i,0], proj[i,1]))
plt.xlabel("PC1")
plt.ylabel("PC2")
st.pyplot(plt)

st.subheader("Weights Heatmap (Agents × Generators)")
weights_matrix = np.array([agent.weights for agent in agents])
plt.figure(figsize=(8,3))
sns.heatmap(weights_matrix, annot=True, cmap="coolwarm", cbar=True)
plt.xlabel("Generators (1-24)")
plt.ylabel("Agents")
st.pyplot(plt)

# -------------------------
# Interactive Counseling
# -------------------------
st.subheader("Causal Counseling")
user_text = st.text_area("Describe your mental state / question:")
if st.button("Get Causal Analysis"):
    counselor = URICounselor()
    analysis = counselor.counsel(user_text)
    st.text_area("Analysis & Suggested Path", value=analysis, height=100)
