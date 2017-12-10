package net.huutonauru.neural;

import java.util.Vector;

public class Backpropagation extends Network {

    Vector<Double> getTotalErrorForOutputs(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        ensureOutputSizeEqualsToExpectedOutputSize(output, expectedOutput);
        return calculateTotalErrorForOutputNeurons(output, expectedOutput);
    }

    Vector<Double> calculateSquaredErrorForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        ensureOutputSizeEqualsToExpectedOutputSize(output, expectedOutput);
        return calculateSquaredErrorForOutputNeurons(output, expectedOutput);
    }

    double calculateSqaredErrorSumForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        ensureOutputSizeEqualsToExpectedOutputSize(output, expectedOutput);
        return calculateSquaredErrorSumForOutputNeurons(output, expectedOutput);
    }

    Vector<Double> calculateErrorForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        ensureOutputSizeEqualsToExpectedOutputSize(output, expectedOutput);
        return calculateErrorForOutputNeurons(output, expectedOutput);
    }

    private void ensureOutputSizeEqualsToExpectedOutputSize(Layer output, double[] expectedOutput) throws NeuralNetworkError {
        if (output.size() != expectedOutput.length) {
            throw new NeuralNetworkError("Output size different from given expected outputs: " + output.size() + " != " + expectedOutput.length);
        }
    }

    private Vector<Double> calculateSquaredErrorForOutputNeurons(Layer output, double[] expectedOutput) {
        Vector<Double> res = new Vector<Double>();
        for (int i = 0; i < output.size(); i++) {
            res.add(calculateSquaredErrorForOutputNeuron(output.get(i), expectedOutput[i]));
        }
        return res;
    }

    private Vector<Double> calculateErrorForOutputNeurons(Layer output, double[] expectedOutput) {
        Vector<Double> res = new Vector<Double>();
        for (int i = 0; i < output.size(); i++) {
            res.add(calculateErrorForOutputNeuron(output.get(i), expectedOutput[i]));
        }
        return res;
    }

    private double calculateTotalErrorForOutputNeuron(Neuron neuron, double expected, double totalSquaredError) {
        double error = calculateErrorForOutputNeuron(neuron, expected);
        double derivate = getPartialLogisticDerivateOfValue(neuron.getValue());
        return error * derivate * totalSquaredError;
    }

    private Vector<Double> calculateTotalErrorForOutputNeurons(Layer output, double[] expectedOutput) {
        Vector<Double> res = new Vector<Double>();

        double totalSquaredError = calculateSquaredErrorSumForOutputNeurons(output, expectedOutput);
        for (int i = 0; i < output.size(); i++) {
            res.add(calculateTotalErrorForOutputNeuron(output.get(i), expectedOutput[i], totalSquaredError));
        }
        return res;
    }

    private double calculateSquaredErrorSumForOutputNeurons(Layer output, double[] expectedOutput) {
        double res = 0.0;
        for (int i = 0; i < output.size(); i++) {
            res += calculateSquaredErrorForOutputNeuron(output.get(i), expectedOutput[i]);
        }
        return res;
    }

    double calculateSquaredErrorForOutputNeuron(Neuron outputNeuron, double expected) {
        return 0.5 * Math.pow(expected - outputNeuron.getValue(), 2);
    }

    double calculateErrorForOutputNeuron(Neuron outputNeuron, double expected) {
        return -(expected - outputNeuron.getValue());
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
