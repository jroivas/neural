package net.huutonauru.neural;

import java.util.Vector;

public class Backpropagation extends Network {

    double squareError = 0;

    void calculateError(Vector<Neuron> output) {
        Vector<Neuron> prev = output;
        squareError = 0;
        for (int i = (int)size() - 2; i >= 1; i--) {
            for (int j = 0; j < get(i).size(); j++) {
                Neuron n = get(i).get(j);
                double sum = 0;
                for (int k = 0; k < prev.size(); k++) {
                    Neuron tmp = prev.elementAt(k);
                    //tmp.getError() * tmp.getLinkFrom(n).;
                }
            }
        }
    }

    double getSquareError() {
        return squareError;
    }
}
