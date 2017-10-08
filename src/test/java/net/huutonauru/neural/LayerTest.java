package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class LayerTest {

    @Test
    public void createLayer() {
        assertNotNull(new Layer());
    }

    @Test
    public void emptyLayer() {
        Layer layer = new Layer();
        assertEquals(layer.size(), 0);
    }

    @Test
    public void createLayerWithNeurons() {
        Layer layer = new Layer();
        layer.addNeuron(new Neuron());
        assertEquals(layer.size(), 1);
    }
}
