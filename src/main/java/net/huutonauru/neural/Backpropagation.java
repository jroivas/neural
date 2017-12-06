package net.huutonauru.neural;

import java.util.Vector;

public class Backpropagation extends Network {

    Vector<Double> calculateErrorForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        if (output.size() != expectedOutput.length) {
            throw new NeuralNetworkError("Output size different from given expected outputs: " + output.size() + " != " + expectedOutput.length);
        }

        return calculateErrorForOutputNeurons(output, expectedOutput);
    }

    double calculateErrorSumForOutput(double[] expectedOutput) throws NeuralNetworkError {
        Layer output = last();
        if (output.size() != expectedOutput.length) {
            throw new NeuralNetworkError("Output size different from given expected outputs: " + output.size() + " != " + expectedOutput.length);
        }

        return calculateErrorSumForOutputNeurons(output, expectedOutput);
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
}
