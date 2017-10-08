package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class LinkTest {

    @Test
    public void createLinkBetweenTwoNeurons() {
        assertNotNull(new Link(new Neuron(), new Neuron()));
    }

    @Test
    public void checkLinkDestinations() {
        Link link = new Link(new Neuron(), new Neuron());
        assertNotNull(link.getFrom());
        assertNotNull(link.getTo());
    }

    @Test
    public void getLinkWeight() {
        Link.setWeightRange(-0.3, 0.3);
        Link link = new Link(new Neuron(), new Neuron());
        double weight = link.getWeight();

        assertTrue(weight >= -0.3);
        assertTrue(weight <= 0.3);
    }

    @Test
    public void getLinkWeightWithCustomWeightRange() {
        Link.setWeightRange(1.0, 2.0);
        Link link = new Link(new Neuron(), new Neuron());
        double weight = link.getWeight();

        assertTrue(weight >= 1.0);
        assertTrue(weight <= 2.0);
    }
}
