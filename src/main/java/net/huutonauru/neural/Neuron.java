package net.huutonauru.neural;

import java.util.Vector;
import java.util.concurrent.ThreadLocalRandom;

import lombok.Getter;
import lombok.Setter;

public class Neuron {

    @Getter @Setter private double value;
    @Getter private Sigmoid sigmoid;
    @Getter @Setter private double error;
    private double rangeMin;
    private double rangeMax;
    private Vector<Link> links = new Vector<Link>();

    public Neuron() {
        this.sigmoid = new DefaultSigmoid();
    }

    public Neuron(double value) {
        this.sigmoid = new DefaultSigmoid();
        this.value = value;
    }

    public Neuron(Sigmoid sigmoid) {
        this.sigmoid = sigmoid;
    }

    public Neuron(double value, Sigmoid sigmoid) {
        this.sigmoid = sigmoid;
        this.value = value;
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

    Link getLinkFrom(Neuron from) {
        for (Link l : links) {
            if (l.getFrom() == from) return l;
        }
        return null;
    }

    long linkCount() {
        return links.size();
    }

    double calculateWeightedSum() {
        double sum = 0;
        for (int i = 0; i < links.size(); i++) {
            sum += links.get(i).calculateWeightedValue();
        }
        return sum;
    }

    void pass() {
        double sum = calculateWeightedSum();
        value = sigmoid.transfer(sum);
    }
}
