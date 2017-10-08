package net.huutonauru.neural;

import java.util.Vector;

public class Network {
    private Vector<Layer> layers = new Vector<Layer>();

    void addLayer(Layer l) {
        layers.add(l);
    }

    long size() {
        return layers.size();
    }

    Layer get(int index) {
        return layers.get(index);
    }

    Layer first() {
        return layers.firstElement();
    }

    Layer last() {
        return layers.lastElement();
    }

    void linkNeuronToLayer(Neuron from, Layer to) {
        for (int i = 0; i < to.size(); i++) {
            from.linkTo(to.get(i));
        }
    }

    void linkLayerToAnother(Layer from, Layer to) {
        for (int i = 0; i < from.size(); i++) {
            linkNeuronToLayer(from.get(i), to);
        }
    }

    void linkAll() {
        for (int i = 0; i < size() - 1; i++) {
            linkLayerToAnother(get(i), get(i+1));
        }
    }
}
