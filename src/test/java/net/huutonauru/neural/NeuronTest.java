package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class NeuronTest {

    @Test
    public void createNewNeuron() {
        Neuron neuron = new Neuron();
        assertNotNull(neuron);
    }

    @Test
    public void NeuronValue() {
        double value = new Neuron().getValue();
        assertEquals(value, 0.0);
        //assertTrue(value >= 0.0);
        //assertTrue(value <= 1.0);
    }

    /*
    @Test
    public void NeuronValueOtherThanZero() {
        double value = new Neuron(new NonZeroSigmoid()).getValue();
        assertTrue(value > 0.0);
        assertTrue(value <= 1.0);
    }

    @Test
    public void NeuronDifferentValues() {
        // FIXME Actually this can fail
        assertTrue(new Neuron().getValue() != new Neuron().getValue());
    }
    */

    @Test
    public void neuronLinkTo() {
        Neuron neuron1 = new Neuron();
        Neuron neuron2 = new Neuron();

        neuron1.linkTo(neuron2);

        assertEquals(neuron1.linkCount(), 0);
        assertEquals(neuron2.linkCount(), 1);
    }

    @Test
    public void neuronLinkFrom() {
        Neuron neuron1 = new Neuron();
        Neuron neuron2 = new Neuron();

        neuron1.linkFrom(neuron2);

        assertEquals(neuron1.linkCount(), 1);
        assertEquals(neuron2.linkCount(), 0);
    }
}
