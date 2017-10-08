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
}
