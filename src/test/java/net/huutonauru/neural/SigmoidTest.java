package net.huutonauru.neural;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

public class SigmoidTest {

    /*
    @Test
    public void DefaultSigmoidSanity() {
        Sigmoid s = new DefaultSigmoid();
        assertTrue(s.getMin() < s.getMax());
    }

    @Test
    public void LogSigmoidSanity() {
        Sigmoid s = new DefaultSigmoid();
        assertTrue(s.getMin() < s.getMax());
    }

    @Test
    public void LogSigmoidMinNotZero() {
        Sigmoid s = new LogSigmoid();
        assertTrue(s.getMin() > 0.0);
    }
    */

    @Test
    public void DefaultSigmoidTranfer() {
        Sigmoid s = new DefaultSigmoid();
        assertEquals(s.transfer(1), 1);
    }

    @Test
    public void LogSigmoidTranfer() {
        Sigmoid s = new LogSigmoid();
        assertNotEquals(s.transfer(1), 1);
    }
}
