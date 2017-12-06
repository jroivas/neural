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

    private Backpropagation createTestNetwork(int outputSize) {
        Backpropagation net = new Backpropagation();
        net.addLayer(new Layer(2));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(outputSize));
        net.linkAll();
        return net;
    }

    private void setNetworkInputValues(Backpropagation net, double[] input) {
        try {
            net.first().setValues(input);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when setting values to layer");
        }
    }

    private Backpropagation newBackpropagationWithForwardPass(double[] input, int outputSize) {
        Backpropagation net = createTestNetwork(outputSize);
        setNetworkInputValues(net, input);
        net.forwardPass();
        return net;
    }

    @Test
    public void forwardPassOnBackpropagation() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        assertTrue(net.last().first().getValue() != 0);
    }

    @Test
    public void calculateErrorForOneOutput() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        Neuron outputNeuron = net.last().first();
        double error = net.calculateErrorForOutputNeuron(outputNeuron, 3.0);
        assertTrue(error != 0);
    }

    @Test
    public void calculateErrorForOutputLayerManually() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);

        Neuron outputNeuron = net.last().first();
        double error1 = net.calculateErrorForOutputNeuron(outputNeuron, 3.0);
        outputNeuron = net.last().last();
        double error2 = net.calculateErrorForOutputNeuron(outputNeuron, 0.5);

        assertTrue(error1 != 0);
        assertTrue(error2 != 0);
        assertTrue(error1 != error2);
    }

    @Test
    public void calculateErrorForOutputLayerWrongSizeOfParameters() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        double[] expectedOutput = {3.0, 5.0};
        Throwable exception = assertThrows(NeuralNetworkError.class, () -> {
            net.calculateErrorForOutput(expectedOutput);
        });
    }

    @Test
    public void calculateErrorForOutputLayer() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        double[] expectedOutput = {3.0};
        Vector<Double> errors = null;
        try {
            errors = net.calculateErrorForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        assertEquals(1, errors.size());
    }

    @Test
    public void calculateErrorForOutputLayerMatchManualCalculation() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        double[] expectedOutput = {3.0};
        Vector<Double> errors = null;
        try {
            errors = net.calculateErrorForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        double error = net.calculateErrorForOutputNeuron(net.last().first(), expectedOutput[0]);
        assertTrue(error == errors.get(0));
    }

    @Test
    public void calculateErrorForOutputLayerWithTwoOutputs() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);

        double[] expectedOutput = {3.0, 0.5};
        Vector<Double> errors = null;
        try {
            errors = net.calculateErrorForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        assertEquals(2, errors.size());
        assertNotEquals(errors.get(0), errors.get(1));
    }

    @Test
    public void calculateErrorSumForOutputLayer() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);

        double[] expectedOutput = {3.0, 0.5};
        double error = 0;
        try {
            error = net.calculateErrorSumForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        assertTrue(error > 0);
    }
}
