package net.huutonauru.neural;

import java.util.concurrent.ThreadLocalRandom;

import lombok.Getter;
import lombok.Setter;

public class Link {

    @Getter private Neuron from;
    @Getter private Neuron to;
    @Getter @Setter private double weight;
    private static double rangeMin = -0.3;
    private static double rangeMax = 0.3;

    Link(Neuron from, Neuron to) {
        this.from = from;
        this.to = to;

        this.weight = ThreadLocalRandom.current().nextDouble(rangeMin, rangeMax);
    }

    static void setWeightRange(double min, double max) {
        rangeMin = min;
        rangeMax = max;
    }

    double calculateWeightedFromValue() {
        return weight * from.getValue();
    }
}
