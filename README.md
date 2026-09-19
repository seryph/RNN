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

The output can then be c
