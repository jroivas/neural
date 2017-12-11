package net.huutonauru.neural;

import java.util.Vector;

public class Layer {
    private Vector<Neuron> neurons;

    Layer() {
        neurons = new Vector<Neuron>();
    }

    Layer(int size) {
        init(size, new DefaultSigmoid());
    }

    Layer(int size, Sigmoid sigmoid) {
        init(size, sigmoid);
    }

    void init(int size, Sigmoid sigmoid) {
        neurons = new Vector<Neuron>();
        generateNeurons(size, sigmoid);
    }

    long size() {
        return neurons.size();
    }

    Neuron get(int index) {
        return neurons.get(index);
    }

    Neuron first() {
        return neurons.firstElement();
    }

    Neuron last() {
        return neurons.lastElement();
    }

    Layer addNeuron(Neuron n) {
        neurons.add(n);
        return this;
    }

    Layer generateNeurons(int num) {
        return generateNeurons(num, new DefaultSigmoid());
    }

    Layer generateNeurons(int num, Sigmoid sigmoid) {
        for (int i = 0; i < num; i++) {
            addNeuron(new Neuron(sigmoid));
        }
        return this;
    }

    private void ensureValuesSizeMatchLayerSize(double[] values) throws NeuralNetworkError {
        if (values.length != size()) {
            throw new NeuralNetworkError("Input size different from given values: " + size() + " != " + values.length);
        }
    }

    void setValues(double[] values) throws NeuralNetworkError {
        ensureValuesSizeMatchLayerSize(values);
        for (int i = 0; i < size(); i++) {
            get(i).setValue(values[i]);
        }
    }

    Vector<Double> getValues() {
        Vector<Double> res = new Vector<Double>();
        for (Neuron neuron : neurons) {
            res.add(neuron.getValue());
        }
        return res;
    }

    Vector<Double> getWeights() {
        Vector<Double> res = new Vector<Double>();
        for (Neuron neuron : neurons) {
            for (Link link : neuron.getLinks()) {
                res.add(link.getWeight());
            }
        }
        return res;
    }

    private void ensureErrorsSizeMatchLayerSize(Vector<Double> errors) throws NeuralNetworkError {
        if (errors.size() != size()) {
            throw new NeuralNetworkError("Layer size different from given errors size: " + size() + " != " + errors.size());
        }
    }

    void adjustWeights(Vector<Double> errors, double learningRate) throws NeuralNetworkError {
        ensureErrorsSizeMatchLayerSize(errors);
        for (int i = 0; i < neurons.size(); i++) {
            for (Link link : neurons.get(i).getLinks()) {
                link.adjustWeight(errors.get(i), learningRate);
            }
        }
    }

    void linkNeuron(Neuron from) {
        for (Neuron neuron : neurons) {
            from.linkTo(neuron);
        }
    }

    void linkToAnother(Layer to) {
        for (Neuron neuron : neurons) {
            to.linkNeuron(neuron);
        }
    }

    void pass() {
        for (Neuron neuron : neurons) {
            neuron.pass();
        }
    }
}
