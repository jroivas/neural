package net.huutonauru.neural;

public interface Sigmoid {
    public double getMin();
    public double getMax();
}

class DefaultSigmoid implements Sigmoid {
    public double getMin() { return 0.0; }
    public double getMax() { return 1.0; }
}

class NonZeroSigmoid implements Sigmoid {
    public double getMin() { return 0.01; }
    public double getMax() { return 1.0; }
}
