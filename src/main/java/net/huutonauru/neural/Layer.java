package net.huutonauru.neural;

import java.util.Vector;

public class Layer {
    private Vector<Neuron> neurons;

    Layer() {
        neurons = new Vector<Neuron>();
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

    void addNeuron(Neuron n) {
        neurons.add(n);
    }

    void generateNeurons(long num) {
        for (long i = 0; i < num; i++) {
            addNeuron(new Neuron());
        }
    }

    void generateNeurons(long num, Sigmoid sigmoid) {
    }
}
