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

    void linkAll() {
    }
}
