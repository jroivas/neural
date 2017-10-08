package net.huutonauru.neural;

public interface Sigmoid {
    public double getMin();
    public double getMax();
    public double transfer(double value);
}

class DefaultSigmoid implements Sigmoid {
    public double getMin() { return 0.0; }
    public double getMax() { return 1.0; }
    public double transfer(double value) {
        return value;
    }
}

class LogSigmoid implements Sigmoid {
    public double getMin() { return 0.01; }
    public double getMax() { return 1.0; }
    private double p1;
    private double p2;
    LogSigmoid(double param1, double param2) {
        p1 = param1;
        p2 = param2;
    }
    public double transfer(double value) {
        return (1 + p1) / (1 + Math.exp(p2 * -value));
    }
}
