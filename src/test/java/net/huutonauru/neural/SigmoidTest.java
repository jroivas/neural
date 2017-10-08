package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class SigmoidTest {

    @Test
    public void DefaultSigmoidSanity() {
        Sigmoid s = new DefaultSigmoid();
        assertTrue(s.getMin() < s.getMax());
    }

    @Test
    public void NonZeroSigmoidSanity() {
        Sigmoid s = new DefaultSigmoid();
        assertTrue(s.getMin() < s.getMax());
    }

    @Test
    public void NonZeroSigmoidMinNotZero() {
        Sigmoid s = new NonZeroSigmoid();
        assertTrue(s.getMin() > 0.0);
    }
}
