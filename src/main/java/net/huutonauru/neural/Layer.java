package net.huutonauru.neural;

import java.util.Vector;

public class Layer {
    Vector<Neuron> neurons;

    Layer() {
        neurons = new Vector<Neuron>();
    }

    long size() {
        return neurons.size();
    }

    void addNeuron(Neuron n) {
        neurons.add(n);
    }
}
