package net.huutonauru.neural;

import java.util.Vector;

public class Layer {
    private Vector<Neuron> neurons;

    public Layer() {
        neurons = new Vector<Neuron>();
    }

    public Layer(int size) {
        init(size, new DefaultSigmoid());
    }

    public Layer(int size, Sigmoid sigmoid) {
        init(size, sigmoid);
    }

    public void init(int size, Sigmoid sigmoid) {
        neurons = new Vector<Neuron>();
        generateNeurons(size, sigmoid);
    }

    public long size() {
        return neurons.size();
    }

    public Neuron get(int index) {
        return neurons.get(index);
    }

    public Neuron first() {
        return neurons.firstElement();
    }

    public Neuron last() {
        return neurons.lastElement();
    }

    public Layer addNeuron(Neuron n) {
        neurons.add(n);
        return this;
    }

    public Layer generateNeurons(int num) {
        return generateNeurons(num, new DefaultSigmoid());
    }

    public Layer generateNeurons(int num, Sigmoid sigmoid) {
        for (int i = 0; i < num; i++) {
            addNeuron(new Neuron(sigmoid));
        }
        return this;
    }

    private void ensureValuesSizeMatchLayerSize(double... values) throws NeuralNetworkError {
        if (values.length != size()) {
            throw new NeuralNetworkError("Input size different from given values: " + size() + " != " + values.length);
        }
    }

    public void setValues(double... values) throws NeuralNetworkError {
        ensureValuesSizeMatchLayerSize(values);
        for (int i = 0; i < size(); i++) {
            get(i).setValue(values[i]);
        }
    }

    public Vector<Double> getValues() {
        Vector<Double> res = new Vector<Double>();
        for (Neuron neuron : neurons) {
            res.add(neuron.getValue());
        }
        return res;
    }

    public Vector<Double> getWeights() {
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

    public void adjustNeuronWeights(Neuron neuron, double error, double learningRate) {
        for (Link link : neuron.getLinks()) {
            link.adjustWeight(error, learningRate);
        }
    }

    public void adjustLayerWeights(Vector<Double> errors, double learningRate) {
        for (int i = 0; i < neurons.size(); i++) {
            adjustNeuronWeights(neurons.get(i), errors.get(i), learningRate);
        }
    }

    public void adjustWeights(Vector<Double> errors, double learningRate) throws NeuralNetworkError {
        ensureErrorsSizeMatchLayerSize(errors);
        adjustLayerWeights(errors, learningRate);
    }

    public void linkNeuron(Neuron from) {
        for (Neuron neuron : neurons) {
            from.linkTo(neuron);
        }
    }

    public void linkToAnother(Layer to) {
        for (Neuron neuron : neurons) {
            to.linkNeuron(neuron);
        }
    }

    public void pass() {
        for (Neuron neuron : neurons) {
            neuron.pass();
        }
    }
}
