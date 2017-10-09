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
        for (int i = 0; i < size() - 1; i++) {
            get(i).linkToAnother(get(i+1));
        }
    }

    void forwardPass() {
        for (int i = 1; i < size(); i++) {
            get(i).pass();
        }
    }
}
