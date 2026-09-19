# Recurrent Neural Networks (RNNs) — A Practical Tutorial

A beginner-friendly tutorial for learning **Recurrent Neural Networks (RNNs)** from the ground up.

This repository explains how RNNs work, the mathematics behind them, how they learn through **Backpropagation Through Time (BPTT)**, and how to build RNN models in Python for tasks such as **time-series forecasting**.

---

## 📖 What Is an RNN?

A **Recurrent Neural Network (RNN)** is a type of neural network designed to process sequential data.

Unlike a standard feedforward neural network, an RNN maintains information from previous inputs using a **hidden state**. This gives the network a form of memory.

RNNs can be used for:

* Time-series forecasting
* Natural language processing
* Text generation
* Sentiment analysis
* Speech recognition
* Sequence classification

---

## 🧠 How an RNN Works

Suppose we have a sequence:

```text
x₁ → x₂ → x₃ → x₄ → ... → xₜ
```

At each time step, the RNN receives:

1. The current input `xₜ`
2. The previous hidden state `hₜ₋₁`

It then calculates a new hidden state:

```text
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
```

The output can then be calculated as:

```text
yₜ = Wₕᵧhₜ + bᵧ
```

Where:

* `xₜ` = input at time step `t`
* `hₜ` = current hidden state
* `hₜ₋₁` = previous hidden state
* `Wₓₕ` = input-to-hidden weights
* `Wₕₕ` = hidden-to-hidden recurrent weights
* `Wₕᵧ` = hidden-to-output weights
* `bₕ`, `bᵧ` = bias terms

The hidden state allows information from previous time steps to influence future predictions.

---

## 🔄 Unrolling an RNN

An RNN can be visualized as the same neural network repeated across time:

```text
          y₁        y₂        y₃        y₄
          ↑         ↑         ↑         ↑
          │         │         │         │
        [RNN] ─── [RNN] ─── [RNN] ─── [RNN]
          ↑         ↑         ↑         ↑
          │         │         │         │
          x₁        x₂        x₃        x₄
```

The important idea is that these are **not four independent networks**.

The same RNN parameters are shared across every time step.

---

## 🔙 Backpropagation Through Time

RNNs are trained using **Backpropagation Through Time (BPTT)**.

The network is first unrolled across the sequence:

```text
x₁ → RNN → h₁
           ↓
x₂ → RNN → h₂
           ↓
x₃ → RNN → h₃
           ↓
x₄ → RNN → h₄
```

The model calculates a loss, and gradients are propagated backward through the sequence.

Conceptually:

```text
∂L/∂W = Σ ∂Lₜ/∂W
```

Because the hidden states depend on previous hidden states, the chain rule repeatedly multiplies gradients across many time steps.

This creates one of the major challenges with basic RNNs.

---

## ⚠️ Vanishing and Exploding Gradients

During BPTT, gradients may repeatedly become smaller or larger.

### Vanishing Gradients

If gradients repeatedly shrink:

```text
0.5 × 0.5 × 0.5 × 0.5 × 0.5 → 0
```

the network struggles to learn long-term dependencies.

### Exploding Gradients

If gradients repeatedly grow:

```text
2 × 2 × 2 × 2 × 2 → 32
```

training can become unstable.

Common solutions include:

* Gradient clipping
* Better weight initialization
* LSTM networks
* GRU networks

---

## 🧩 LSTM and GRU Networks

Traditional RNNs often struggle to remember information over long sequences.

Two popular alternatives were developed to address this problem:

### Long Short-Term Memory (LSTM)

LSTMs introduce gates that control how information moves through the network.

The major components include:

* Forget gate
* Input gate
* Output gate
* Cell state

### Gated Recurrent Unit (GRU)

GRUs use a simpler architecture with:

* Reset gate
* Update gate

GRUs often provide similar performance to LSTMs while requiring fewer parameters.

---

## 💻 Simple RNN Example

Using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(
        32,
        activation="tanh",
        input_shape=(10, 1)
    ),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()
```

The input shape follows:

```text
(samples, time_steps, features)
```

For example:

```text
(1000, 10, 1)
```

means:

* 1,000 sequences
* 10 time steps per sequence
* 1 feature per time step

---

## 📈 Time-Series Forecasting

One practical application of RNNs is predicting future values from historical observations.

Suppose we have:

```text
10, 12, 14, 16, 18, 20, 22...
```

We can construct sequences such as:

```text
Input               Target

10, 12, 14, 16  →    18
12, 14, 16, 18  →    20
14, 16, 18, 20  →    22
```

The RNN learns the relationship between previous observations and the next value.

---

## 📂 Repository Structure

```text
rnn-tutorial/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_rnn_basics.ipynb
│   ├── 02_rnn_from_scratch.ipynb
│   ├── 03_time_series_rnn.ipynb
│   ├── 04_lstm.ipynb
│   └── 05_gru.ipynb
│
├── src/
│   ├── simple_rnn.py
│   ├── time_series.py
│   ├── lstm.py
│   └── gru.py
│
├── data/
│
└── images/
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd rnn-tutorial
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
jupyter
```

---

## 🎯 Learning Path

A good order for completing this tutorial is:

```text
1. Neural Network Review
        ↓
2. Sequential Data
        ↓
3. RNN Architecture
        ↓
4. RNN Forward Propagation
        ↓
5. Backpropagation Through Time
        ↓
6. Vanishing & Exploding Gradients
        ↓
7. Time-Series Forecasting
        ↓
8. LSTM Networks
        ↓
9. GRU Networks
```

By the end of the tutorial, you should understand both **why RNNs work** and **how to implement them in practice**.

---

## 🧪 Suggested Projects

Once you understand the fundamentals, try building:

* Stock-price sequence predictor
* Weather forecasting model
* Energy-demand forecasting model
* Character-level text generator
* Sentiment classifier
* Sales forecasting model

> **Note:** Forecasting financial markets is inherently uncertain. A model that fits historical data well does not necessarily predict future prices accurately.

---

## 🔑 Key Takeaways

RNNs are designed for **sequential data**.

Their defining feature is the **hidden state**, which allows information from previous time steps to affect future calculations.

However, basic RNNs can suffer from **vanishing and exploding gradients**, particularly with long sequences.

Architectures such as **LSTM** and **GRU** were designed to improve the network's ability to learn longer-term relationships.

Understanding basic RNNs provides an important foundation for studying more advanced sequence models and modern deep-learning architectures.


---

## 📚 Topics Covered

`RNN` `Deep Learning` `Neural Networks` `Time Series` `BPTT` `LSTM` `GRU` `TensorFlow` `Keras` `Python` `Machine Learning`

---

## 📄 License

This project is intended for educational purposes. Add the license of your choice, such as the MIT License, before distributing or accepting contributions.
