package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class NeuronTest {

    @Test
    public void createNewNeuronClass() {
        Neuron neuron = new Neuron();
        assertNotNull(neuron);
    }

    @Test
    public void getNeuronWeight() {
        Neuron neuron = new Neuron();

        double weight = neuron.getWeigth();
        assertTrue(weight >= 0.0);
        assertTrue(weight <= 1.0);
    }
}
