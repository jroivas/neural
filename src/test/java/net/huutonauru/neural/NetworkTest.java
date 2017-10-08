package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class NetworkTest {

    @Test
    public void createNetwork() {
        assertNotNull(new Network());
    }

    @Test
    public void addLayer() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        assertEquals(net.size(), 1);
    }

    @Test
    public void addLayers() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        net.addLayer(new Layer(16));
        assertEquals(net.size(), 2);
    }

    @Test
    public void getLayer() {
        Network net = new Network();
        net.addLayer(new Layer(3));

        assertEquals(net.get(0).size(), 3);
    }

    @Test
    public void getFirstLayer() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        net.addLayer(new Layer(5));

        assertEquals(net.first().size(), 3);
    }

    @Test
    public void getLastLayer() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        net.addLayer(new Layer(5));

        assertEquals(net.last().size(), 5);

        net.addLayer(new Layer(1));
        assertEquals(net.last().size(), 1);
    }


    @Test
    public void getLayers() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        net.addLayer(new Layer(5));

        assertEquals(net.get(1).size(), 5);
    }

    @Test
    public void linkAllLayers() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(1));
        net.linkAll();

        assertEquals(net.size(), 3);
        assertEquals(net.get(0).get(0).linkCount(), 0);
        assertEquals(net.get(1).get(0).linkCount(), 3);
        assertEquals(net.get(2).get(0).linkCount(), 16);
    }
}
