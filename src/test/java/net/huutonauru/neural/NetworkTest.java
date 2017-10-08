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
    public void linkAllLayers() {
        Network net = new Network();
        net.addLayer(new Layer(3));
        net.addLayer(new Layer(16));
        net.addLayer(new Layer(1));
        net.linkAll();

        assertEquals(net.size(), 3);
    }
}
