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

    void setValues(double[] values) throws NeuralNetworkError {
        if (values.length != size()) {
            throw new NeuralNetworkError("Input size different from given values: " + size() + " != " + values.length);
        }
        for (int i = 0; i < size(); i++) {
            get(i).setValue(values[i]);
        }
    }

    void linkNeuron(Neuron from) {
        for (int i = 0; i < size(); i++) {
            from.linkTo(get(i));
        }
    }

    void linkToAnother(Layer to) {
        for (int i = 0; i < this.size(); i++) {
            to.linkNeuron(get(i));
        }
    }

    void pass() {
        for (int i = 0; i < this.size(); i++) {
            get(i).pass();
        }
    }
}
