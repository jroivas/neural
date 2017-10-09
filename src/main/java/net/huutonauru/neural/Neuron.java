package net.huutonauru.neural;

import java.util.Vector;
import java.util.concurrent.ThreadLocalRandom;

import lombok.Getter;
import lombok.Setter;

public class Neuron {

    @Getter @Setter private double value;
    @Getter private Sigmoid sigmoid;
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

    double calculateWeightedSum() {
        double sum = 0;
        for (int i = 0; i < links.size(); i++) {
            Link link = links.get(i);
            sum += link.getFrom().getValue() * link.getWeight();
        }
        return sum;
    }

    void pass() {
        double sum = calculateWeightedSum();
        value = sigmoid.transfer(sum);
    }
}
