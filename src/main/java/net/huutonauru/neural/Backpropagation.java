package net.huutonauru.neural;

import java.util.Vector;

public class Backpropagation extends Network {

    Vector<Double> calculateErrorForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        ensureOutputSizeEqualsToExpectedOutputSize(output, expectedOutput);
        return calculateErrorForOutputNeurons(output, expectedOutput);
    }

    double calculateErrorSumForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        ensureOutputSizeEqualsToExpectedOutputSize(output, expectedOutput);
        return calculateErrorSumForOutputNeurons(output, expectedOutput);
    }

    private void ensureOutputSizeEqualsToExpectedOutputSize(Layer output, double[] expectedOutput) throws NeuralNetworkError {
        if (output.size() != expectedOutput.length) {
            throw new NeuralNetworkError("Output size different from given expected outputs: " + output.size() + " != " + expectedOutput.length);
        }
    }

    private Vector<Double> calculateErrorForOutputNeurons(Layer output, double[] expectedOutput) {
        Vector<Double> res = new Vector<Double>();
        for (int i = 0; i < output.size(); i++) {
            res.add(calculateErrorForOutputNeuron(output.get(i), expectedOutput[i]));
        }

        return res;
    }

    private double calculateErrorSumForOutputNeurons(Layer output, double[] expectedOutput) {
        double res = 0.0;
        for (int i = 0; i < output.size(); i++) {
            res += calculateErrorForOutputNeuron(output.get(i), expectedOutput[i]);
        }

        return res;
    }

    double calculateErrorForOutputNeuron(Neuron outputNeuron, double expected) {
        return 0.5 * Math.pow(expected - outputNeuron.getValue(), 2);
    }

    double getPartialLogisticDerivateOfValue(double value) {
        return value * (1 - value);
    }

    Vector<Double> getPartialLogisticDerivateOfOutput() {
        Layer output = last();
        Vector<Double> res = new Vector<Double>();
        for (Double value : output.getValues()) {
            res.add(getPartialLogisticDerivateOfValue(value));
        }
        return res;
    }

}
