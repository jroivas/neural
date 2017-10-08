package net.huutonauru.neural;

import java.util.Vector;
import java.util.concurrent.ThreadLocalRandom;

import lombok.Getter;
import lombok.Setter;

public class Neuron {

    @Getter private double value;
    @Getter private Sigmoid sigmoid;
    private double rangeMin;
    private double rangeMax;
    private Vector<Link> links = new Vector<Link>();

    public Neuron() {
        this.sigmoid = new DefaultSigmoid();
        init();
    }

    public Neuron(Sigmoid sigmoid) {
        this.sigmoid = sigmoid;
        init();
    }

    private void init() {
        //value = ThreadLocalRandom.current().nextDouble(sigmoid.getMin(), sigmoid.getMax());
    }

    void linkTo(Neuron to) {
        to.addLink(new Link(this, to));
    }

    void linkFrom(Neuron from) {
        addLink(new Link(from, this));
    }

    void addLink(Link link) {
        links.add(link);
    }

    long linkCount() {
        return links.size();
    }
}
