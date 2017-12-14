package net.huutonauru.neural;

import java.util.Vector;

public class Network {
    private Vector<Layer> layers = new Vector<Layer>();

    public void addLayer(Layer l) {
        layers.add(l);
    }

    public long size() {
        return layers.size();
    }

    public Layer get(int index) {
        return layers.get(index);
    }

    public Layer first() {
        return layers.firstElement();
    }

    public Layer last() {
        return layers.lastElement();
    }

    public void linkAll() {
        for (int i = 0; i < size() - 1; i++) {
            get(i).linkToAnother(get(i+1));
        }
    }

    public void forwardPass() {
        for (int i = 1; i < size(); i++) {
            get(i).pass();
        }
    }
}
