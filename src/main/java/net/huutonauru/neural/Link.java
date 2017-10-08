package net.huutonauru.neural;

import lombok.Getter;

public class Link {

    @Getter private Neuron from;
    @Getter private Neuron to;

    Link(Neuron from, Neuron to) {
        this.from = from;
        this.to = to;
    }
}
