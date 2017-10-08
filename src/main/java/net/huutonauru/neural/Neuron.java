package net.huutonauru.neural;

import java.util.concurrent.ThreadLocalRandom;

import lombok.Getter;
import lombok.Setter;

public class Neuron {

    @Getter private double value;
    @Getter private Sigmoid sigmoid;
    private double rangeMin;
    private double rangeMax;

    public Neuron() {
        this.sigmoid = new DefaultSigmoid();
        init();
    }

    public Neuron(Sigmoid sigmoid) {
        this.sigmoid = sigmoid;
        init();
    }

    private void init() {
        value = ThreadLocalRandom.current().nextDouble(sigmoid.getMin(), sigmoid.getMax());
    }
}
