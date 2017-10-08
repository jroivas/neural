package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.util.NoSuchElementException;

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
    public void createLayerAddNeuron() {
        Layer layer = new Layer();

        layer.addNeuron(new Neuron());

        assertEquals(layer.size(), 1);
    }

    @Test
    public void createLayerAddNeurons() {
        Layer layer = new Layer();

        layer.addNeuron(new Neuron());
        layer.addNeuron(new Neuron());

        assertEquals(layer.size(), 2);
    }

    @Test
    public void createLayerWithNeurons() {
        Layer layer = new Layer();
        layer.generateNeurons(10);

        assertEquals(layer.size(), 10);
    }

    @Test
    public void createLayerWithNeuronsConstructor() {
        assertEquals(new Layer(10).size(), 10);
    }

    @Test
    public void createLayerWithNeuronssSameLayer() {
        Layer layer = new Layer();
        assertEquals(layer, layer.generateNeurons(10));
    }

    @Test
    public void createLayerAddNeuronsSameLayer() {
        Layer layer = new Layer();
        assertEquals(layer, layer.addNeuron(new Neuron()));
    }

    @Test
    public void getNeuronAtFirstIndex() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertNotNull(layer.get(0));
    }

    @Test
    public void getNeuronFirst() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertNotNull(layer.first());
    }

    @Test
    public void getNeuronFirstOnEmptyLayerFails() {
        Layer layer = new Layer();
        Throwable exception = assertThrows(NoSuchElementException.class, () -> {
            layer.first();
        });
    }

    @Test
    public void getNeuronLastOnEmptyLayerFails() {
        Layer layer = new Layer();
        Throwable exception = assertThrows(NoSuchElementException.class, () -> {
            layer.last();
        });
    }

    @Test
    public void getNeuronLast() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertNotNull(layer.last());
    }

    @Test
    public void getNeuronFistSameFirstByIndex() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertEquals(layer.first(), layer.get(0));
    }

    @Test
    public void getNeuronFistNotSameAsLast() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertNotEquals(layer.first(), layer.last());
    }

    @Test
    public void getNeuronAtSecondIndex() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertNotNull(layer.get(1));
    }

    @Test
    public void getNeuronOutsideRange() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        Throwable exception = assertThrows(ArrayIndexOutOfBoundsException.class, () -> {
            layer.get(10);
        });
        assertEquals(exception.getMessage(), "Array index out of range: 10");
    }

    @Test
    public void getNeuronsSame() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertEquals(layer.get(0), layer.get(0));
    }

    @Test
    public void getNeuronsNotSame() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertNotEquals(layer.get(0), layer.get(1));
    }

    @Test
    public void generateNeuronsGetSigmoid() {
        Layer layer = new Layer();
        layer.generateNeurons(3);

        assertTrue(layer.first().getSigmoid() instanceof DefaultSigmoid);
    }

    @Test
    public void addNeuronWithNonZeroSigmoid() {
        Layer layer = new Layer();
        layer.addNeuron(new Neuron(new NonZeroSigmoid()));

        assertFalse(layer.first().getSigmoid() instanceof DefaultSigmoid);
        assertTrue(layer.first().getSigmoid() instanceof NonZeroSigmoid);
    }

    @Test
    public void createLayerWithSigmoidNeurons() {
        Layer layer = new Layer();
        layer.generateNeurons(10, new NonZeroSigmoid());

        assertEquals(layer.size(), 10);
        assertTrue(layer.first().getSigmoid() instanceof NonZeroSigmoid);
    }

    @Test
    public void createLayerWithNeuronsConstructorAndDefaultSigmoid() {
        Layer layer = new Layer(10);
        assertEquals(layer.size(), 10);
        assertTrue(layer.first().getSigmoid() instanceof DefaultSigmoid);
    }

    @Test
    public void createLayerWithNeuronsConstructorAndNonZeroSigmoid() {
        Layer layer = new Layer(10, new NonZeroSigmoid());
        assertEquals(layer.size(), 10);
        assertTrue(layer.first().getSigmoid() instanceof NonZeroSigmoid);
    }

    @Test
    public void setValuesToLayer() {
        Layer layer = new Layer(5);
        double[] input = {1.0, 2, 30, 42, 11};
        try {
            layer.setValues(input);
        }
        catch (NeuralNetworkError e) {
            assertNull("Exception thrown!");
        }

        assertEquals(layer.get(0).getValue(), 1.0);
        assertEquals(layer.get(1).getValue(), 2);
        assertEquals(layer.get(2).getValue(), 30);
        assertEquals(layer.get(3).getValue(), 42);
        assertEquals(layer.get(4).getValue(), 11);
    }
}
