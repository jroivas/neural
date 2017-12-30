package net.huutonauru.neural;

public interface Sigmoid {
    public double transfer(double value);
}

class DefaultSigmoid implements Sigmoid {
    public double transfer(double value) {
        return value;
    }
}

class LogSigmoid implements Sigmoid {
    private final double p1;
    private final double p2;
    public LogSigmoid(double param1, double param2) {
        p1 = param1;
        p2 = param2;
    }
    public double transfer(double value) {
        return (1 + p1) / (1 + Math.exp(p2 * -value));
    }
}
