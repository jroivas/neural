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
    public void setLearningRate() {
        Backpropagation p = new Backpropagation();
        double rate = p.getLearningRate();

        p.setLearningRate(3.14159);
        assertNotEquals(rate, p.getLearningRate());
        assertTrue(p.getLearningRate() == 3.14159);
    }

    @Test
    public void forwardPassOnBackpropagation() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        assertTrue(net.last().first().getValue() != 0);
    }

    @Test
    public void calculateSquaredErrorForOneOutput() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        Neuron outputNeuron = net.last().first();
        double error = net.calculateSquaredErrorForOutputNeuron(outputNeuron, 3.0);
        assertTrue(error != 0);
    }

    @Test
    public void calculateErrorForOneOutput() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        Neuron outputNeuron = net.last().first();
        outputNeuron.setValue(5.0);
        double error = net.calculateErrorForOutputNeuron(outputNeuron, 3.0);
        assertTrue(error == 2.0);
    }

    @Test
    public void calculateNegativeErrorForOneOutput() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        Neuron outputNeuron = net.last().first();
        outputNeuron.setValue(2.0);
        double error = net.calculateErrorForOutputNeuron(outputNeuron, 3.0);
        assertTrue(error == -1.0);
    }

    @Test
    public void calculateSquaredErrorForOutputLayerManually() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);

        Neuron outputNeuron = net.last().first();
        double error1 = net.calculateSquaredErrorForOutputNeuron(outputNeuron, 3.0);
        outputNeuron = net.last().last();
        double error2 = net.calculateSquaredErrorForOutputNeuron(outputNeuron, 0.5);

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
            net.calculateSqaredErrorSumForOutput(expectedOutput);
        });
    }

    @Test
    public void calculateSquaredErrorForOutputLayer() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        double[] expectedOutput = {3.0};
        Vector<Double> errors = getSquaredErrorFromExpectedOutputAsListOfDouble(net, expectedOutput);
        assertEquals(1, errors.size());
    }

    @Test
    public void calculateSquaredErrorForOutputLayerMatchManualCalculation() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        double[] expectedOutput = {3.0};
        Vector<Double> errors = getSquaredErrorFromExpectedOutputAsListOfDouble(net, expectedOutput);
        double error = net.calculateSquaredErrorForOutputNeuron(net.last().first(), expectedOutput[0]);
        assertTrue(error == errors.get(0));
    }

    @Test
    public void calculateSquaredErrorForOutputLayerWithTwoOutputs() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);

        double[] expectedOutput = {3.0, 0.5};
        Vector<Double> errors = getSquaredErrorFromExpectedOutputAsListOfDouble(net, expectedOutput);
        assertEquals(2, errors.size());
        assertNotEquals(errors.get(0), errors.get(1));
    }

    @Test
    public void calculateErrorForOutputLayer() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);

        double[] expectedOutput = {3.0, 0.5};
        Vector<Double> errors = getErrorFromExpectedOutputAsListOfDouble(net, expectedOutput);
        assertEquals(2, errors.size());
        assertNotEquals(errors.get(0), errors.get(1));
        assertTrue(errors.get(0) == -(3.0 - net.last().first().getValue()) );
    }

    @Test
    public void calculateErrorSumForOutputLayer() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);
        double[] expected = {3.0, 0.5};
        double error = getErrorFromExpectedOutputAsDouble(net, expected);
        assertTrue(error > 0);
    }

    @Test
    public void partialDerivateOfValue() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        double derivate = net.getPartialLogisticDerivateOfValue(0.75);
        assertEquals(0.1875, derivate);
    }

    @Test
    public void partialDerivateOfValueZero() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        double derivate = net.getPartialLogisticDerivateOfValue(0);
        assertEquals(0, derivate);
    }

    @Test
    public void partialDerivateOfValueOne() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        double derivate = net.getPartialLogisticDerivateOfValue(1);
        assertEquals(0, derivate);
    }

    @Test
    public void partialDerivateOfValueHalf() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        double derivate = net.getPartialLogisticDerivateOfValue(0.5);
        assertEquals(0.25, derivate);
    }

    @Test
    public void partialDerivateOfOutput() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        Vector<Double> derivates = net.getPartialLogisticDerivateOfOutput();
        assertEquals(1, derivates.size());
        assertTrue(derivates.get(0) == net.getPartialLogisticDerivateOfValue(net.last().first().getValue()));
    }

    @Test
    public void partialDerivateOfManyOutputs() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);
        Vector<Double> derivates = net.getPartialLogisticDerivateOfOutput();
        assertEquals(2, derivates.size());
        assertTrue(derivates.get(0) == net.getPartialLogisticDerivateOfValue(net.last().first().getValue()));
        assertTrue(derivates.get(1) == net.getPartialLogisticDerivateOfValue(net.last().last().getValue()));
    }

    @Test
    public void totalErrorForOutputs() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        double[] expected = {3.0};
        checkTotalErrorMatchValues(net, expected);
    }

    @Test
    public void totalErrorForMultipleOutputs() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 2);
        double[] expected = {3.0, 0.5};
        checkTotalErrorMatchValues(net, expected);
    }

    @Test
    public void changeWeightsToOuputLayerWithTotalError() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        double[] expected = {3.0};
        Vector<Double> totalErrors = getTotalErrorFromExpectedOutputAsListOfDouble(net, expected);

        Vector<Double> weights = net.last().getWeights();
        assertEquals(16, weights.size());

        try {
            net.adjustLayerWeights(net.last(), totalErrors);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when adjusting weights");
        }

        Vector<Double> weights2 = net.last().getWeights();
        assertNotEquals(weights.get(0), weights2.get(0));
        assertNotEquals(weights.get(1), weights2.get(1));
        assertTrue(weights.get(0) - net.getLearningRate() * totalErrors.get(0) == weights2.get(0));
        assertTrue(weights.get(1) - net.getLearningRate() * totalErrors.get(0) == weights2.get(1));
        assertTrue(weights.get(1) - net.getLearningRate() * totalErrors.get(0) != weights2.get(0));
    }

    @Test
    public void calculateDerivatedError() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);
        Neuron outputNeuron = net.last().first();

        double res = net.calculateDerivateErrorForOutputNeuron(outputNeuron, 0.5);
        double error = net.calculateErrorForOutputNeuron(outputNeuron, 0.5);
        double derivate = net.getPartialLogisticDerivateOfValue(outputNeuron.getValue());
        assertEquals(res, error * derivate);
    }

    @Test
    public void calculateDerivatedErrors() {
        double[] input = {1, 2};
        Backpropagation net = newBackpropagationWithForwardPass(input, 1);

        double[] expected = {3.0};
        Vector<Double> res = getDerivateErrorFromExpectedOutputAsListOfDouble(net, expected);

        Neuron outputNeuron = net.last().first();
        double error = net.calculateErrorForOutputNeuron(outputNeuron, 3.0);
        double derivate = net.getPartialLogisticDerivateOfValue(outputNeuron.getValue());
        assertTrue(res.get(0) == error * derivate);
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

    private Vector<Double> getSquaredErrorFromExpectedOutputAsListOfDouble(Backpropagation net, double[] expectedOutput) {
        try {
            return net.calculateSquaredErrorForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        return null;
    }

    private Vector<Double> getTotalErrorFromExpectedOutputAsListOfDouble(Backpropagation net, double[] expectedOutput) {
        try {
            return net.getTotalErrorForOutputs(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when setting values to layer");
        }
        return null;
    }

    private Vector<Double> getDerivateErrorFromExpectedOutputAsListOfDouble(Backpropagation net, double[] expectedOutput) {
        try {
            return net.getDerivateErrorForOutputs(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when setting values to layer");
        }
        return null;
    }

    private Vector<Double> getErrorFromExpectedOutputAsListOfDouble(Backpropagation net, double[] expectedOutput) {
        try {
            return net.calculateErrorForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        return null;
    }

    private double getErrorFromExpectedOutputAsDouble(Backpropagation net, double[] expectedOutput) {
        try {
            return net.calculateSqaredErrorSumForOutput(expectedOutput);
        }
        catch (NeuralNetworkError e) {
            fail("Exception thrown when calculating output");
        }
        return 0;
    }

    private void checkTotalErrorMatchValues(Backpropagation net, double[] expected) {
        Vector<Double> totalErrors = getTotalErrorFromExpectedOutputAsListOfDouble(net, expected);
        assertEquals(expected.length, totalErrors.size());

        Vector<Double> errors = getErrorFromExpectedOutputAsListOfDouble(net, expected);
        Vector<Double> derivates = net.getPartialLogisticDerivateOfOutput();
        double error = getErrorFromExpectedOutputAsDouble(net, expected);

        for (int i = 0; i < expected.length; i++) {
            assertTrue(totalErrors.get(i) == errors.get(i) * derivates.get(i) * error);
        }
    }

}
