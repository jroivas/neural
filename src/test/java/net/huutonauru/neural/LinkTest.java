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
}
