import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import scipy.io
from scipy import sparse
from keras.models import Sequential
from keras.layers.core import Dense
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import numpy as np
#import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.models import load_model
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import sys
import time






class Reservoir(object):
    """
    Build a reservoir and evaluate internal states

    Parameters:
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
        leak = amount of leakage in the reservoir state update (optional)
        connectivity = percentage of nonzero connection weights (unused in circle reservoir)
        input_scaling = scaling of the input connection weights
        noise_level = deviation of the Gaussian noise injected in the state update
        circle = generate determinisitc reservoir with circle topology
    """

    def __init__(self, n_internal_units=100, spectral_radius=0.99, leak=None,
                 connectivity=0.3, input_scaling=0.2, noise_level=0.01, circle=False):

        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                n_internal_units,
                spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius)

    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):

        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0, -1] = spectral_radius
        for i in range(n_internal_units - 1):
            internal_weights[i + 1, i] = spectral_radius

        return internal_weights

    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()  # cria uma matriz esparca

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)  # calcula o autovalor
        e_max = np.max(np.abs(E))  # calcula o autovalor maximo
        internal_weights /= np.abs(e_max) / spectral_radius

        return internal_weights

    def _compute_state_matrix(self, X, n_drop=0):  # n_drop => transiente
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        state_matrix = np.empty((N, T - n_drop, self._n_internal_units), dtype=float)
        for t in range(T):
            current_input = X[:, t, :]

            # Calculate state
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

            # Add noise
            state_before_tanh += np.random.rand(self._n_internal_units, N) * self._noise_level

            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak) * previous_state + np.tanh(state_before_tanh).T

            # Armazene tudo após o período de abandono
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix

    def get_states(self, X, n_drop=0, bidir=True):
        N, T, V = X.shape
        if self._input_weights is None:  # calcula os pesos de entrada
            self._input_weights = (2.0 * np.random.binomial(1, 0.5,
                                                            [self._n_internal_units, V]) - 1.0) * self._input_scaling

        # compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop)

        # estados do reservatório na entrada invertida no tempo
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states

    def getReservoirEmbedding(self, X, pca, ridge_embedding, n_drop=5, bidir=True, test=False):

        res_states = self.get_states(X, n_drop=5, bidir=True)

        N_samples = res_states.shape[0]
        res_states = res_states.reshape(-1, res_states.shape[2])
        # ..transform..
        if test:
            red_states = pca.transform(res_states)
        else:
            red_states = pca.fit_transform(res_states)
            # ..and put back in tensor form
        red_states = red_states.reshape(N_samples, -1, red_states.shape[1])

        coeff_tr = []
        biases_tr = []

        for i in range(X.shape[0]):
            ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
            coeff_tr.append(ridge_embedding.coef_.ravel())
            biases_tr.append(ridge_embedding.intercept_.ravel())
        st.text('Setting the parameters...')
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        return input_repr


pca = PCA(n_components=100)
ridge_embedding = Ridge(alpha=10, fit_intercept=True)
ESN = Reservoir(n_internal_units=200, spectral_radius=0.9698384,
                leak=0.76073533, connectivity=0.25, input_scaling=0.9, noise_level=0.0011135, circle=False)
ini = time.time()
st.title("      Cardiac series analysis")
st.image('coracao.png')
st.markdown("This app is able to classify through a patient's cardiac time series, whether the patient is healthy, mildly ill or has a severe disease")
st.subheader("Select the sample")

control = pd.read_csv("C.csv");
control = control.values;
control = control.T;
control = control[:, :1000];
df = pd.DataFrame(control);
df = df.fillna(df.mean());
control = df.values;

leprosy = pd.read_csv("h.csv");
leprosy = leprosy.values;
leprosy = leprosy.T;
leprosy = leprosy[:, :1000];
df1 = pd.DataFrame(leprosy);
df1 = df1.fillna(df1.mean());
leprosy = df1.values;

bipolar = pd.read_csv("b_a.csv");
bipolar1 = pd.read_csv("b_b.csv");
bipolar = bipolar.values;
bipolar = bipolar.T;
bipolar1 = bipolar1.values;
bipolar1 = bipolar1.T;
bipolar1 = bipolar1[:, :1000];
bipolar = bipolar[:, :1000];
bipolar = np.vstack([bipolar, bipolar1]);
df2 = pd.DataFrame(bipolar);
df2 = df2.fillna(df2.mean());
bipolar = df2.values;
irc = pd.read_csv("I.csv");
irc = irc.values;
irc = irc.T;
irc = irc[:, :1000];
df3 = pd.DataFrame(irc);
df3 = df3.fillna(df3.mean());
irc = df3.values;
brain_death = pd.read_csv("MG.csv")
brain_death = brain_death.values
brain_death = brain_death.T
brain_death = np.vstack([brain_death, brain_death, brain_death, brain_death, brain_death, brain_death, brain_death])
df4 = pd.DataFrame(brain_death)
df4 = df4.fillna(df4.mean())
brain_death = df4.values
uti = pd.read_csv("u.csv")
uti = uti.values
uti = uti.T
uti1 = uti[:, :1000]
uti2 = uti[:10, 2000:3000]
UTI = np.vstack([uti1, uti2]);
df5 = pd.DataFrame(UTI)
df5 = df5.fillna(df5.mean())
UTI = df5.values
data = np.vstack([control, leprosy, bipolar, irc, brain_death, UTI])
st.text('Loading data...!')
Y1 = np.zeros(control.shape[0], dtype=int)
Y1.shape
Y2 = np.zeros(leprosy.shape[0], dtype=int)
Y2
Y3 = np.zeros(bipolar.shape[0], dtype=int)
Y3.shape
Y4 = np.zeros(irc.shape[0], dtype=int) + 1
Y4.shape
Y5 = np.zeros(brain_death.shape[0], dtype=int) + 1
Y5.shape
Y6 = np.zeros(UTI.shape[0], dtype=int) + 1
Y6.shape
y = np.hstack([Y1, Y2, Y3, Y4, Y5, Y6])
indice = np.arange(data.shape[0])
rng = np.random.RandomState(123)  # vamos embrulhar os indices
permutacao_indice = rng.permutation(indice)
X = data
Y = y
train_size = int(0.65 * X.shape[0])
test_size = X.shape[0] - (train_size)  # indices para testes
train_ind = permutacao_indice[:train_size]
value_ind = permutacao_indice[train_size:train_size]
test_ind = permutacao_indice[train_size:]
X_train, Y_train = X[train_ind], Y[train_ind]
X_test, Y_test = X[test_ind], Y[test_ind]
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
Y_train = Y_train.reshape(Y_train.shape[0], -1)
Y_test = Y_test.reshape(Y_test.shape[0], -1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
onehot_encoder = OneHotEncoder(sparse=False)
Y_train_hot = onehot_encoder.fit_transform(Y_train)
Y_test_hot = onehot_encoder.fit_transform(Y_test)
import streamlit as st

amostra = st.file_uploader('Select the sample (.txt)', type = 'txt')
if amostra is not None:
    st.text('"PROCESSING ..."')
    input_repr = ESN.getReservoirEmbedding(X_train, pca, ridge_embedding, n_drop=5, bidir=True, test=False)
    input_repr_te = ESN.getReservoirEmbedding(X_test, pca, ridge_embedding, n_drop=5, bidir=True, test=True)

    model = Sequential()
    model.add(Dense(500, input_dim=10100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(Y_train_hot.shape[1], activation='softmax'))

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    history = model.fit(input_repr, Y_train_hot, epochs=30, shuffle=True, batch_size=32, verbose=0)

    acuracia = model.evaluate(input_repr_te, Y_test_hot)
    amostra = np.loadtxt(amostra)
    amostra = np.hstack([amostra, amostra])
    amostra = amostra.reshape(amostra.shape[0], -1)
    amostra = amostra.T
    amostra = amostra[:, :1000]
    amostra = (amostra - mu) / sigma
    amostra = amostra.reshape(1, 1000, -1)
    amostra = ESN.getReservoirEmbedding(amostra, pca, ridge_embedding, n_drop=5, bidir=True, test=True)
    clas = (model.predict(amostra) > 0.5).astype("int32")
    clas = onehot_encoder.inverse_transform(clas)
    clas = clas.reshape(-1)
    if clas == 0:
        control = pd.read_csv("C.csv")
        control = control.values
        control = control.T
        control = control[:, :1000]
        df = pd.DataFrame(control)
        df = df.fillna(df.mean())
        control = df.values
        leprosy = pd.read_csv("h.csv")
        leprosy = leprosy.values
        leprosy = leprosy.T
        leprosy = leprosy[:, :1000]
        df1 = pd.DataFrame(leprosy)
        df1 = df1.fillna(df1.mean())
        leprosy = df1.values
        bipolar = pd.read_csv("b_a.csv")
        bipolar1 = pd.read_csv("b_b.csv")
        bipolar = bipolar.values
        bipolar = bipolar.T
        bipolar1 = bipolar1.values
        bipolar1 = bipolar1.T
        bipolar1 = bipolar1[:, :1000]
        bipolar = bipolar[:, :1000]
        bipolar = np.vstack([bipolar, bipolar1])
        df2 = pd.DataFrame(bipolar)
        df2 = df2.fillna(df2.mean())
        bipolar = df2.values
        data1 = np.vstack([control, leprosy, bipolar])

        Y1 = np.zeros(control.shape[0], dtype=int)

        Y2 = np.zeros(leprosy.shape[0], dtype=int) + 1

        Y3 = np.zeros(bipolar.shape[0], dtype=int) + 1

        y = np.hstack([Y1, Y2, Y3])

        indice = np.arange(data1.shape[0])
        rng = np.random.RandomState(123)  # vamos embrulhar os indices
        permutacao_indice = rng.permutation(indice)
        X1 = data1
        Y = y
        train_size = int(0.65 * X1.shape[0])
        test_size = X1.shape[0] - (train_size)  # indices para testes
        train_ind = permutacao_indice[:train_size]
        value_ind = permutacao_indice[train_size:train_size]
        test_ind = permutacao_indice[train_size:]
        X_train, Y_train = X1[train_ind], Y[train_ind]

        X_test, Y_test = X1[test_ind], Y[test_ind]

        mu1, sigma1 = X_train.mean(axis=0), X_train.std(axis=0)
        X_train = (X_train - mu1) / sigma1
        X_test = (X_test - mu1) / sigma1

        Y_train = Y_train.reshape(Y_train.shape[0], -1)
        Y_test = Y_test.reshape(Y_test.shape[0], -1)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

        onehot_encoder = OneHotEncoder(sparse=False)
        Y_train_hot = onehot_encoder.fit_transform(Y_train)
        Y_test_hot = onehot_encoder.fit_transform(Y_test)

        pca = PCA(n_components=100)
        ridge_embedding = Ridge(alpha=10, fit_intercept=True)
        ESN = Reservoir(n_internal_units=200, spectral_radius=0.9698384,
                        leak=0.76073533, connectivity=0.25, input_scaling=0.9, noise_level=0.0011135, circle=False)
        input_repr = ESN.getReservoirEmbedding(X_train, pca, ridge_embedding, n_drop=5, bidir=True, test=False)
        input_repr_te = ESN.getReservoirEmbedding(X_test, pca, ridge_embedding, n_drop=5, bidir=True, test=True)

        model = Sequential()
        model.add(Dense(500, input_dim=10100, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(400, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation="relu"))
        model.add(Dense(Y_train_hot.shape[1], activation='softmax'))

        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
        history = model.fit(input_repr, Y_train_hot, epochs=30, shuffle=True, batch_size=32, verbose=0)

        model.evaluate(input_repr_te, Y_test_hot)
        model.predict(amostra)
        clas = (model.predict(amostra) > 0.5).astype("int32")

        clas = onehot_encoder.inverse_transform(clas)
        clas = clas.reshape(-1)
        acuracia = model.evaluate(input_repr_te, Y_test_hot)
        if clas == 0:
            st.subheader("The class of sample: 'Healthy' ")
        elif clas == 1:
            st.subheader("The class of sample:  'Mild diseases'")
        clas1 = (model.predict(amostra)).astype("float32")
        clas1 = clas1.reshape(-1)
        a = clas1[0]
        b = clas1[1]

        st.write("Likely to be healthy(%)", a.astype("float32") * 100)
        st.write("Likely to be Mild diseases(%)", b.astype("float32") * 100)
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect="equal"))
        labels = "healthy(%)", "Mild diseases(%)"
        c = [clas1[0], clas1[1]]
        total = sum(c)
        ax.pie(c, autopct='%0.0001f%%')
        ax.legend(labels, title="Likely to be", loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
        ax.set_title("Prediction Graph")
        st.pyplot(fig)


        st.write("Algorithm prediction(%)", acuracia[1] * 100)


        fim = time.time()
        st.write("runtime(s)", fim - ini)
       # sys.exit()
    elif clas == 1:

        clas1 = (model.predict(amostra)).astype("float32")
        clas1 = clas1.reshape(-1)
        st.subheader("The class of sample: 'Severe diseases' ")
        a = clas1[0]
        b = clas1[1]
        st.write("Likely to be healthy(%)", a.astype("float32") * 100)
        st.write("Likely to be Severe diseases(%)", b.astype("float32") * 100)
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect="equal"))
        labels = "healthy(%)", "Severe diseases(%)"
        c = [clas1[0],clas1[1]]
        total = sum(c)
        ax.pie(c, autopct='%0.0001f%%')
        ax.legend(labels, title="Likely to be", loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
        ax.set_title("Prediction Graph")
        st.pyplot(fig)




        st.write("Algorithm prediction(%)", acuracia[1] * 100)

        fim = time.time()
        st.write("runtime(s)", fim - ini)