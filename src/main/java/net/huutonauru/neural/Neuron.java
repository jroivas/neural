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
    @Getter private Vector<Link> links = new Vector<Link>();

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

    public void linkTo(Neuron to) {
        to.addLink(new Link(this, to));
    }

    public void linkFrom(Neuron from) {
        addLink(new Link(from, this));
    }

    public void addLink(Link link) {
        links.add(link);
    }

    public Link getLinkFrom(Neuron from) {
        for (Link l : links) {
            if (l.getFrom() == from) return l;
        }
        return null;
    }

    public long linkCount() {
        return links.size();
    }

    public double calculateWeightedSum() {
        double sum = 0;
        for (int i = 0; i < links.size(); i++) {
            sum += links.get(i).calculateWeightedValue();
        }
        return sum;
    }

    public void pass() {
        double sum = calculateWeightedSum();
        value = sigmoid.transfer(sum);
    }
}
