# Artificial Neural Network library

[Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) library written with Java.

- Written with Java
- Developed TDD style
- Generic forward pass
- Supports [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for training
  * Work in progress still, not fully completed


## Example usage

```
package examples;

import net.huutonauru.neural.Backpropagation;
import net.huutonauru.neural.Layer;
import net.huutonauru.neural.NeuralNetworkError;
import net.huutonauru.neural.Network;

class BackpropagationExample {

    private static Backpropagation createNetwork() {
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

    private static boolean setInputValues(Backpropagation net) {
        double[] input = {0.1, 0.8};
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

    private static void printOutputValues(Backpropagation net) {
        // Last layer == output, first == first neuron on layer
        System.out.println("Output neuron value: " + net.last().first().getValue());
    }

    public static void main(String[] args) {
        Backpropagation net = createNetwork();
        if (!setInputValues(net)) return;
        net.forwardPass();
        printOutputValues(net);
    }
}
```

## Dependencies

 - Java 8
 - Maven

## Building

    mvn compile

Or to build package:

    mvn package

## Tests

    mvn test

## Static code analysis

    mvn spotbugs:check

## License

MIT

See [LICENSE](LICENSE) for more info.
