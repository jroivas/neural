package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import java.util.Vector;

public class BackpropagationTest {

    @Test
    public void createBackpropagation() {
        Backpropagation p = new Backpropagation();
    }

    @Test
    public void forwardPass() {
        Backpropagation net = new Backpropagation();
        net.addLayer(new Layer(2));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(1));
        net.linkAll();

        double[] input = {1, 2};
        try {
            net.first().setValues(input);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when setting values to layer");
        }

        net.forwardPass();
        assertTrue(net.last().first().getValue() != 0);
    }

    @Test
    public void calculateError() {
        Backpropagation net = new Backpropagation();
        net.addLayer(new Layer(2));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(1));
        net.linkAll();

        double[] input = {1, 2};
        try {
            net.first().setValues(input);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when setting values to layer");
        }

        net.forwardPass();
        assertTrue(net.last().first().getValue() != 0);

        Vector<Neuron> res = new Vector<Neuron>();
        res.add(new Neuron(3.0));

        double preError = net.getSquareError();

        net.calculateError(res);

        assertTrue(net.getSquareError() != preError);
    }
}
