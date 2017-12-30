package examples;

import net.huutonauru.neural.Backpropagation;
import net.huutonauru.neural.Layer;
import net.huutonauru.neural.NeuralNetworkError;

class BackpropagationExample {

    public static void main(String[] args) {
        BackpropagationExampleHelper helper = new BackpropagationExampleHelper();
        double[] input = {0.1, 0.8};
        if (!helper.setInputValues(input)) return;
        helper.forwardPass();
        helper.printOutputValues();
    }
}

class BackpropagationExampleHelper {
    Backpropagation net;

    BackpropagationExampleHelper() {
        net = createNetwork();
    }

    private Backpropagation createNetwork() {
        /* Create network with:
         * 2 input neurons,
         * 8 neurons on first hidden layer
         * 6 neurons on second hidden layer
         * 1 neuron on output layer
         * Fully link
         */
        Backpropagation net = new Backpropagation();
        net.addLayer(new Layer(2));
        net.addLayer(new Layer(8));
        net.addLayer(new Layer(6));
        net.addLayer(new Layer(1));
        net.linkAll();
        return net;
    }

    public boolean setInputValues(double... input) {
        try {
            // First layer is input layer
            net.first().setValues(input);
        }
        catch (NeuralNetworkError e) {
            System.err.println("Couldn't set input values to network!");
            return false;
        }
        return true;
    }

    public void printOutputValues() {
        // Last layer == output, first == first neuron on layer
        System.out.println("Output neuron value: " + net.last().first().getValue());
    }

    public void forwardPass() {
        net.forwardPass();
    }

}

